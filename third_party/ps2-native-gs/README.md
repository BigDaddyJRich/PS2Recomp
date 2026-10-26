# ps2-native-gs

Renderizador nativo do **GS (Graphics Synthesizer)** do PlayStation 2, feito para **recompilações estáticas** de jogos
de PS2. Em vez de emular o chip gráfico, ele traduz o estado do GS em chamadas de desenho da GPU do computador:
**Metal** no macOS e **Vulkan** no Windows, com o mesmo código de renderização nos dois.

> 🚧 **Projeto em evolução constante.** O ps2-native-gs está sempre sendo aprimorado: precisão, desempenho e
> recursos novos chegam a cada jogo recompilado com ele. Espere mudanças frequentes.

Nasceu na recompilação de *Silent Hill Origins* e é o renderizador padrão dos próximos recomps.

## O que ele faz

- **Desenha em alta resolução** (1× a 8× a resolução interna do jogo) direto na janela; a imagem nunca volta para a
  CPU.
- **Reproduz o GS com fidelidade**: mistura (alpha blending) programável, teste de alfa, máscara de escrita (FBMASK),
  névoa, filtro bilinear com os pesos do próprio GS, regiões de textura (CLAMP/REGION_REPEAT), texturas indexadas
  (paletas) e leituras do próprio alvo (efeitos de pós-processamento dos jogos).
- **Alvos de renderização na GPU**: o que o jogo desenha e depois lê como textura fica na GPU, inclusive leituras com
  outro formato (visões de 16 bits de um alvo de 32 bits, profundidade lida como cor, cópias do canal alfa).
- **Shaders especializados** por combinação de estado, compilados em segundo plano.
- **Substituição de texturas** compatível com os pacotes HD do PCSX2 (mesmos hashes).
- **Efeitos de imagem modernos, todos opcionais** (desligados por padrão): nitidez adaptativa, oclusão de ambiente e
  SMAA (ver abaixo).
- **Ferramentas de validação**: grava a entrada do GS de uma cena (gs-dump) e a reproduz em qualquer módulo
  (`gs-replay`), comparando quadro a quadro.

## Arquitetura

```
src/native_gs_module.mm   o renderizador: decodifica o GS e decide o que desenhar (independente de API)
src/gpu.h                 a interface de GPU que ele usa: texturas, passes, cópias, pipelines, swapchain
src/gpu_metal.mm          implementação em Metal (macOS)
src/gpu_vulkan.cpp        implementação em Vulkan (Windows; no macOS também roda via MoltenVK, para testes)
src/gs_shaders_msl.inc    shaders em MSL (Metal)
src/gs_shaders.glsl       os mesmos shaders em GLSL (Vulkan); manter os dois em sincronia
src/compile_shaders.sh    GLSL -> SPIR-V embutido (gs_shaders_spv.inc)
src/gs_decode.h           front end do GS: registradores, pacotes GIF, transferências para a VRAM
tools/gs_replay.cpp       reprodução e comparação de gs-dumps
```

O módulo é uma biblioteca carregada pelo runtime do recomp por uma ABI pequena (`include/runtime/gs/gs_parallel_api.h`,
versão 6, ponto de entrada `black_parallel_gs_api`): o runtime entrega pacotes GIF, escritas de registradores e
transferências, e pede a imagem de cada quadro (`scanout`) ou que ela vá direto para a janela (`attach_window`).

### Metal × Vulkan

O ponto delicado é a **mistura programável**: o GS mistura cores com fórmulas que o blending fixo das GPUs de PC não
reproduz, então o shader lê o pixel que está desenhando.

- **Metal**: leitura direta do framebuffer (`[[color(0)]]`), em ordem, nativa nas GPUs Apple.
- **Vulkan**: o alvo é também *input attachment*. Com `VK_EXT_rasterization_order_attachment_access` (AMD, entre
  outras) a ordem é garantida pela GPU; sem ela (NVIDIA, Intel), o desenho é dividido em grupos de triângulos que não
  se sobrepõem, com uma barreira entre grupos (teste exato de sobreposição; uma malha 3D vira um grupo só).
- **Mistura fixa** (GPUs sem a extensão de ordem): toda draw cuja fórmula do GS cabe no blending da GPU usa o blending
  fixo com *dual-source blending* (o shader calcula `|ks|·Cs` e o fator `C/128`), sem ler o alvo e sem barreiras; só
  as fórmulas que não cabem continuam lendo o destino. Entre 7 e 87 vezes mais rápido nas cenas de teste via
  MoltenVK; a diferença para o caminho exato é de arredondamento (1 a 3 tons, por camada). `PS2X_GS_EXACT_BLEND=1`
  volta ao caminho exato; no Metal, `PS2X_GS_FIXED_BLEND=1` força a mistura fixa, para comparar.
- `VK_EXT_extended_dynamic_state` é usado quando existe; sem ele, o estado de profundidade entra no pipeline.

O backend Vulkan foi validado contra o Metal pixel a pixel (684 quadros de 7 cenas, em 1× e 4×, via MoltenVK).

## Efeitos de imagem

Os efeitos não mudam nada com eles desligados; cada um é ligado à parte.

- **A cena 3D separada do 2D.** Os efeitos não devem atingir o que o jogo desenha por cima do 3D (granulado de filme,
  vinhetas, HUD, menus). O renderizador acompanha cada quadro e, quando o jogo passa de desenhos com teste de
  profundidade para desenhos sem ele, tira uma cópia do alvo na GPU. A última cópia do quadro é o 3D antes do 2D.
  Os efeitos são calculados sobre essa cópia, e só a mudança que eles fazem é somada à imagem final: proporcional
  onde o 2D só escureceu a cena (fades, vinhetas) e nula onde o 2D a substituiu (HUD). Não depende do jogo: é a regra
  de profundidade do próprio GS. A estrutura do quadro anterior diz qual transição é a última, para copiar uma vez só.
- **Nitidez** (`PS2X_GS_NATIVE_SHARPEN=0..1`): realce adaptativo ao contraste, no estilo do CAS da AMD.
- **Oclusão de ambiente** (`PS2X_GS_NATIVE_AO=0..1`): sombra nos cantos e onde objetos se encostam, só a partir do
  buffer de profundidade, sem conhecer a projeção do jogo. O GS interpola o Z de forma linear na tela, então numa
  superfície plana a média de dois pontos opostos em volta de um pixel é o Z do próprio pixel; onde a média fica à
  frente, há uma dobra côncava. Pares que atravessam um salto de profundidade (silhuetas) são ignorados. É calculada
  uma vez por quadro, na resolução do jogo, e ampliada respeitando a profundidade (*joint bilateral upsample*).
- **SMAA 1x** (`PS2X_GS_NATIVE_SMAA=1`, no lugar do FXAA): porte da implementação de referência (preset ULTRA, sem
  detecção diagonal). Uma borda só conta se aparece na imagem final e na cópia da cena: o granulado não vira borda e
  o HUD não é borrado.

Diagnóstico com o `gs-replay`: `PS2X_GS_NATIVE_SCANOUT_PRESENT=1` devolve a imagem como a janela a recebe (com os
efeitos), `PS2X_GS_NATIVE_SCANOUT_SCENE=1` a cópia da cena, e com `PS2X_GS_NATIVE_SCANOUT_AO=1` só a oclusão.

## Compilar

Requisitos: CMake 3.20+, Ninja, um compilador C++20 (Clang/Xcode no macOS; llvm-mingw ou MSVC/Clang no Windows).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

- **macOS**: gera `libps2-native-gs.so` (Metal), `libps2-native-gs-vk.so` (Vulkan via MoltenVK, para testes) e
  `gs-replay`. `-DPS2_NATIVE_GS_VULKAN=OFF` dispensa o Vulkan.
- **Windows**: gera `libps2-native-gs.dll` (Vulkan). Precisa de um driver com Vulkan 1.1.
- O CMake baixa `volk` e `Vulkan-Headers` (FetchContent). Para usar cópias locais:
  `-DFETCHCONTENT_SOURCE_DIR_VOLK=... -DFETCHCONTENT_SOURCE_DIR_VULKAN_HEADERS=...`.

### Shaders Vulkan

O SPIR-V já vem pronto em `src/gs_shaders_spv.inc`. Depois de mudar `src/gs_shaders.glsl`:

```bash
src/compile_shaders.sh   # precisa do glslangValidator (Vulkan SDK ou `brew install glslang`)
```

## Usar num recomp

O runtime carrega o módulo pelo caminho em `PS2X_GS_PARALLEL_MODULE`. As opções mais usadas:

| Variável | Efeito |
|---|---|
| `PS2X_GS_NATIVE_SCALE=1..8` | resolução interna (múltiplo da do jogo) |
| `PS2X_GS_NATIVE_PRESENT=1` | desenha direto na janela (Metal layer / janela-filha Vulkan) |
| `PS2X_GS_NATIVE_GAMMA`, `PS2X_GS_NATIVE_BRIGHTNESS` | ajuste de imagem na saída |
| `PS2X_GS_NATIVE_FXAA=0` | desliga a suavização de bordas (FXAA) |
| `PS2X_GS_NATIVE_SMAA=1` | SMAA no lugar do FXAA |
| `PS2X_GS_NATIVE_SHARPEN=0..1` | nitidez (0 = desligada) |
| `PS2X_GS_NATIVE_AO=0..1`, `PS2X_GS_NATIVE_AO_RADIUS` | oclusão de ambiente (0 = desligada) e o raio, em pixels do jogo |
| `PS2X_TEXTURE_PACKS=<pasta>` | pacotes de texturas HD (formato PCSX2) |
| `PS2X_GS_DUMP=<arquivo>` + `PS2X_GS_DUMP_REQUEST=<arquivo>` | grava uma cena para o `gs-replay` |
| `PS2X_GS_MOLTENVK=<libMoltenVK.dylib>` | (macOS) Vulkan via MoltenVK |
| `PS2X_GS_VULKAN_VALIDATION=1` | liga a camada de validação do Vulkan |

Há outras chaves de diagnóstico (`PS2X_GS_NATIVE_*`, `PS2X_GS_VULKAN_*`) descritas nos comentários do código.

## Validar com gs-replay

```bash
# reproduz uma cena e salva os quadros
build/gs-replay cena.gsdump --module build/libps2-native-gs.so --out quadros-metal --every 10
build/gs-replay cena.gsdump --module build/libps2-native-gs-vk.so --out quadros-vulkan --every 10
# compara
build/gs-replay --diff quadros-metal quadros-vulkan
```

Toda mudança no renderizador deve sair com 0 quadros diferentes nas cenas de referência, a não ser que a diferença seja
justamente a correção pretendida.

## Licença

GPL-3.0 (ver `LICENSE`). Inclui código do [PS2Recomp](https://github.com/ran-j/PS2Recomp) (layout de memória do GS), o
xxHash (BSD 2-Clause) e as tabelas do [SMAA](https://github.com/iryoku/smaa) (MIT, Jimenez et al.), em `src/third_party`;
os passos do SMAA nos shaders são um porte da implementação de referência. Nenhum dado de jogo (ELF, ISO, dumps) faz parte deste repositório.
