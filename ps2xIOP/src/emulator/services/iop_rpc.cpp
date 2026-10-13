#include "iop_rpc.h"

#include "../core/iop_cpu.h"
#include "../core/iop_kernel.h"
#include "../core/iop_memory.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <set>
#include <vector>
#include <sstream>
#include <limits>
#include <vector>

namespace ps2x::iop::detail
{
    IopRpcBridge::IopRpcBridge(IopHost &host, IopMemory &memory, IopKernel &kernel) noexcept
        : m_host(host), m_memory(memory), m_kernel(kernel)
    {
    }

    void IopRpcBridge::reset()
    {
        m_servers.clear();
        m_cmdHandlers.clear();
        m_cmdBuffer = 0u;
        m_sysCmdBuffer = 0u;
        m_cmdPacket = 0u;
        m_nextDmaId = 1u;
        m_sifInitialized = false;
    }

    bool IopRpcBridge::dispatchSifManImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 4: // sceSifDma2Init
        case 5: // sceSifInit
            m_sifInitialized = true;
            setV0(0u);
            return true;
        case 7: // sceSifSetDma
        {
            constexpr uint32_t kDescriptorSize = 16u;
            constexpr uint32_t kMaxDescriptors = 32u;
            const uint32_t descriptorAddress = cpu.gpr[4];
            const uint32_t descriptorCount = cpu.gpr[5];
            if (descriptorAddress == 0u || descriptorCount == 0u || descriptorCount > kMaxDescriptors)
            {
                setV0(0u);
                return true;
            }

            struct PendingTransfer
            {
                uint32_t source = 0u;
                uint32_t destination = 0u;
                uint32_t size = 0u;
            };

            std::array<uint32_t, kMaxDescriptors * 4u> descriptorWords{};
            const size_t descriptorBytes = static_cast<size_t>(descriptorCount) * kDescriptorSize;
            if (!m_memory.readRam(descriptorAddress, descriptorWords.data(), descriptorBytes))
            {
                setV0(0u);
                return true;
            }

            std::array<PendingTransfer, kMaxDescriptors> pending{};
            uint32_t pendingCount = 0u;
            uint32_t largestTransfer = 0u;
            for (uint32_t i = 0u; i < descriptorCount; ++i)
            {
                const uint32_t source = descriptorWords[i * 4u + 0u];
                const uint32_t destination = descriptorWords[i * 4u + 1u];
                const int32_t signedSize = static_cast<int32_t>(descriptorWords[i * 4u + 2u]);
                if (signedSize <= 0)
                    continue;

                const uint32_t size = static_cast<uint32_t>(signedSize);
                if (!m_memory.isRamRange(source, size))
                {
                    setV0(0u);
                    return true;
                }
                pending[pendingCount++] = {source, destination, size};
                largestTransfer = std::max(largestTransfer, size);
            }

            // IOP-side sceSifSetDma sends IOP RAM to the EE. Validate all EE
            // destinations before committing any write so a bad chain cannot
            // partially update guest memory, but maybe we could skip this check if we trust the EE-side SIF driver to validate the chain ?!
            // TODO check later
            std::vector<uint8_t> scratch(largestTransfer);
            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_host.readGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            for (uint32_t i = 0u; i < pendingCount; ++i)
            {
                const PendingTransfer &transfer = pending[i];
                if (!m_memory.readRam(transfer.source, scratch.data(), transfer.size) || !m_host.writeGuest(transfer.destination, scratch.data(), transfer.size))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
            {
                m_nextDmaId = 1u;
            }
            setV0(dmaId);
            return true;
        }
        case 8: // sceSifDmaStat
            setV0(0xFFFFFFFFu);
            return true;
        case 29: // sceSifCheckInit
            setV0(m_sifInitialized ? 1u : 0u);
            return true;
        default:
            setV0(0u);
            return true;
        }
    }

    bool IopRpcBridge::dispatchSifCmdImport(uint16_t ordinal, IopCpuState &cpu)
    {
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };
        switch (ordinal)
        {
        case 8: // sceSifSetCmdBuffer(db, size) -> previous buffer
        {
            const uint32_t previous = m_cmdBuffer;
            m_cmdBuffer = cpu.gpr[4];
            setV0(previous);
            return true;
        }
        case 9: // sceSifSetSysCmdBuffer(db, size) -> previous buffer
        {
            const uint32_t previous = m_sysCmdBuffer;
            m_sysCmdBuffer = cpu.gpr[4];
            setV0(previous);
            return true;
        }
        case 10: // sceSifAddCmdHandler(cid, handler, harg)
            m_cmdHandlers[cpu.gpr[4]] = CmdHandler{cpu.gpr[5], cpu.gpr[6], cpu.gpr[28]};
            setV0(0);
            return true;
        case 11: // sceSifRemoveCmdHandler(cid)
            m_cmdHandlers.erase(cpu.gpr[4]);
            setV0(0);
            return true;
        case 4: // InitCmd
        case 5:
        case 6:
        case 7:
        case 14: // InitRpc
            setV0(0);
            return true;
        case 15: // sceSifBindRpc(cd, sid, mode) - the IRX is a client of a server on the EE
        {
            const uint32_t cd = cpu.gpr[4];
            const uint32_t sid = cpu.gpr[5];
            uint32_t sd = 0u, buf = 0u, cbuf = 0u;
            const bool found = m_host.findEeRpcServer(sid, sd, buf, cbuf);
            if (cd != 0u && m_memory.isRamRange(cd, 0x28u))
            {
                m_memory.write32(cd + 0x0Cu, cpu.gpr[6]); // hdr.mode
                m_memory.write32(cd + 0x10u, 0u);         // command
                m_memory.write32(cd + 0x14u, buf);
                m_memory.write32(cd + 0x18u, cbuf);
                m_memory.write32(cd + 0x1Cu, 0u);         // end_function
                m_memory.write32(cd + 0x20u, 0u);         // end_param
                m_memory.write32(cd + 0x24u, sd);         // server (0 = not registered yet; the IRX retries)
            }
            if (const char *dumpPath = std::getenv("HCTP_IOPDUMP"); dumpPath && found)
            {
                static bool dumped = false; // TEMP diag: IOP RAM image when an IRX first binds an EE server
                if (!dumped)
                {
                    dumped = true;
                    if (FILE *f = std::fopen(dumpPath, "wb"))
                    {
                        std::fwrite(m_memory.ram().data(), 1, m_memory.ram().size(), f);
                        std::fclose(f);
                    }
                }
            }
            static std::set<uint64_t> logged;
            if (logged.insert((uint64_t(sid) << 1) | (found ? 1u : 0u)).second)
                std::fprintf(stderr, "[iop-eerpc] IOP bind sid 0x%x -> %s (sd 0x%x buf 0x%x)\n", sid,
                             found ? "EE server" : "none yet", sd, buf);
            setV0(0);
            return true;
        }
        case 16: // sceSifCallRpc(cd, fno, mode, send, ssize, recv, rsize, end_func, end_param) to an EE server
        {
            const uint32_t cd = cpu.gpr[4];
            const uint32_t fno = cpu.gpr[5];
            const uint32_t mode = cpu.gpr[6];
            const uint32_t send = cpu.gpr[7];
            const uint32_t sp = cpu.gpr[29];
            const uint32_t ssize = m_memory.read32(sp + 16u);
            const uint32_t recv = m_memory.read32(sp + 20u);
            const uint32_t rsize = m_memory.read32(sp + 24u);
            const uint32_t endFunction = m_memory.read32(sp + 28u);
            const uint32_t sd = (cd != 0u && m_memory.isRamRange(cd, 0x28u)) ? m_memory.read32(cd + 0x24u) : 0u;
            const uint32_t eeBuf = sd != 0u ? m_memory.read32(cd + 0x14u) : 0u;
            static uint32_t callLogs = 0u;
            if (callLogs < 40u)
            {
                ++callLogs;
                std::fprintf(stderr, "[iop-eerpc] IOP call sd 0x%x fno 0x%x mode %u send 0x%x+0x%x recv 0x%x+0x%x end 0x%x\n",
                             sd, fno, mode, send, ssize, recv, rsize, endFunction);
            }
            if (sd == 0u)
            {
                setV0(static_cast<uint32_t>(-1));
                return true;
            }
            if (send != 0u && ssize != 0u && eeBuf != 0u && m_memory.isRamRange(send, ssize))
            {
                std::vector<uint8_t> bytes(ssize);
                if (m_memory.readRam(send, bytes.data(), ssize))
                    (void)m_host.writeGuest(eeBuf, bytes.data(), ssize);
            }
            const bool blocking = (mode & 1u) == 0u; // SIF_RPC_M_NOWAIT = 1
            const int waiter = blocking ? m_kernel.currentThreadId() : 0;
            IopMemory &memory = m_memory;
            IopKernel &kernel = m_kernel;
            IopHost &host = m_host;
            const bool queued = m_host.callEeRpcServer(sd, fno, ssize,
                [&memory, &kernel, &host, recv, rsize, waiter](uint32_t result)
                {
                    if (recv != 0u && rsize != 0u && result != 0u && memory.isRamRange(recv, rsize))
                    {
                        std::vector<uint8_t> bytes(rsize);
                        if (host.readGuest(result, bytes.data(), rsize))
                            (void)memory.writeRam(recv, bytes.data(), rsize);
                    }
                    if (waiter != 0)
                        kernel.wakeSleeping(waiter);
                });
            if (endFunction != 0u)
            {
                static bool warned = false;
                if (!warned)
                {
                    warned = true;
                    std::fprintf(stderr, "[iop-eerpc] end_function 0x%x not called (unsupported)\n", endFunction);
                }
            }
            setV0(queued ? 0u : static_cast<uint32_t>(-1));
            if (queued && blocking && waiter != 0)
                m_kernel.sleepCurrent(cpu);
            return true;
        }
        case 12: // sceSifSendCmd
        case 13: // isceSifSendCmd
        {
            constexpr uint32_t kHeaderSize = 16u;
            constexpr uint32_t kMaxPacketSize = 112u;
            const uint32_t commandId = cpu.gpr[4];
            const uint32_t packetAddress = cpu.gpr[5];
            const uint32_t packetSize = cpu.gpr[6];
            const uint32_t extraSource = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            const uint32_t extraDestination = m_memory.read32(stackPointer + 16u);
            const int32_t signedExtraSize = static_cast<int32_t>(m_memory.read32(stackPointer + 20u));

            if (packetAddress == 0u || packetSize < kHeaderSize || packetSize > kMaxPacketSize ||
                !m_memory.isRamRange(packetAddress, packetSize))
            {
                setV0(0u);
                return true;
            }

            std::array<uint8_t, kMaxPacketSize> packet{};
            if (!m_memory.readRam(packetAddress, packet.data(), packetSize))
            {
                setV0(0u);
                return true;
            }

            uint32_t extraSize = 0u;
            if (signedExtraSize > 0)
            {
                extraSize = static_cast<uint32_t>(signedExtraSize);
                if (extraSource == 0u || extraDestination == 0u ||
                    !m_memory.isRamRange(extraSource, extraSize) ||
                    !m_host.writeGuest(extraDestination, m_memory.ram().data() + IopMemory::physicalAddress(extraSource), extraSize))
                {
                    setV0(0u);
                    return true;
                }
            }

            const uint32_t sizeWord = packetSize | (extraSize << 8u);
            std::memcpy(packet.data() + 0u, &sizeWord, sizeof(sizeWord));
            std::memcpy(packet.data() + 4u, &extraDestination, sizeof(extraDestination));
            std::memcpy(packet.data() + 8u, &commandId, sizeof(commandId));

            if (!m_host.sendSifCommand(commandId, packet.data(), packetSize))
            {
                // A command without an EE handler is still a completed DMA on  real hardware. Only malformed packets fail above.
            }

            const uint32_t dmaId = m_nextDmaId++;
            if (m_nextDmaId == 0u || m_nextDmaId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
                m_nextDmaId = 1u;
            setV0(dmaId);
            return true;
        }
        case 17: // sceSifRegisterRpc
        {
            RpcServer server;
            server.serverData = cpu.gpr[4];
            server.sid = cpu.gpr[5];
            server.function = cpu.gpr[6];
            server.gp = cpu.gpr[28];
            server.buffer = cpu.gpr[7];
            const uint32_t stackPointer = cpu.gpr[29];
            server.callback = m_memory.read32(stackPointer + 16u);
            server.callbackBuffer = m_memory.read32(stackPointer + 20u);
            server.queue = m_memory.read32(stackPointer + 24u);
            m_servers[server.sid] = server;
            if (server.serverData != 0u)
            {
                m_memory.write32(server.serverData + 0x20u, server.sid);
                m_memory.write32(server.serverData + 0x28u, server.function);
                m_memory.write32(server.serverData + 0x2Cu, server.buffer);
            }
            setV0(server.serverData);
            return true;
        }
        case 18:
            setV0(0);
            return true;
        case 19: // SetRpcQueue
            setV0(cpu.gpr[4]);
            return true;
        case 20:
        case 21:
            setV0(0);
            return true;
        case 22: // RpcLoop
            m_kernel.sleepCurrent(cpu);
            setV0(0);
            return true;
        case 23: // sceSifGetOtherData(rd, src (EE), dest (IOP), size, mode): the IOP pulls EE memory over SIF DMA.
        {
            // HCTP's yiop sound IRX fetches its sound banks this way; the stub returned without copying -> empty banks,
            // no sound effects (ADX music, which streams by another path, still played). Done at once, so a NOWAIT
            // caller finds the transfer finished.
            const uint32_t src = cpu.gpr[5];
            const uint32_t dest = cpu.gpr[6] & 0x1FFFFFu;
            const int32_t size = static_cast<int32_t>(cpu.gpr[7]);
            bool ok = size > 0 && m_memory.isRamRange(dest, static_cast<uint32_t>(size));
            if (ok)
            {
                std::vector<uint8_t> scratch(static_cast<size_t>(size));
                ok = m_host.readGuest(src, scratch.data(), scratch.size()) &&
                     m_memory.writeRam(dest, scratch.data(), scratch.size());
            }
            static uint32_t logged = 0u;
            if (logged++ < 32u)
                std::fprintf(stderr, "[iop-getother] src=0x%x dest=0x%x size=%d ok=%d\n", src, dest, size, ok ? 1 : 0);
            setV0(ok ? 0u : static_cast<uint32_t>(-1));
            return true;
        }
        case 24: // RemoveRpc
        {
            const uint32_t serverData = cpu.gpr[4];
            for (auto server = m_servers.begin(); server != m_servers.end(); ++server)
            {
                if (server->second.serverData == serverData)
                {
                    m_servers.erase(server);
                    break;
                }
            }
            setV0(0);
            return true;
        }
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
            setV0(0);
            return true;
        default:
            return false;
        }
    }

    RpcResult IopRpcBridge::handleRpc(const RpcRequest &request, IopGuestExecutor &executor)
    {
        RpcResult result{};
        const auto serverIt = m_servers.find(request.sid);
        if (serverIt == m_servers.end() || serverIt->second.function == 0u)
            return result;

        RpcServer &server = serverIt->second;
        static const bool trace = std::getenv("VENDETTA_IOPLOG") != nullptr;
        if (trace)
        {
            std::ostringstream out;
            out << "[iop-rpc] sid=0x" << std::hex << request.sid << " fn=0x" << request.function
                << " send=0x" << request.send.address << "+0x" << request.send.size
                << " recv=0x" << request.receive.address << "+0x" << request.receive.size;
            m_host.log(LogLevel::Info, out.str());
        }
        if (request.send.size != 0u && server.buffer != 0u)
        {
            const uint32_t copySize = std::min<uint32_t>(request.send.size, IopMemory::RamSize - std::min(server.buffer, IopMemory::RamSize));
            if (copySize != 0u)
            {
                std::vector<uint8_t> payload(copySize);
                if (m_host.readGuest(request.send.address, payload.data(), payload.size()))
                    (void)m_memory.writeRam(server.buffer, payload.data(), payload.size());
            }
        }

        uint32_t returnPointer = executor.executeGuestFunction(server.function,
                                                               request.function,
                                                               server.buffer,
                                                               request.send.size,
                                                               0u,
                                                               server.gp);
        if (returnPointer == 0u)
            returnPointer = server.buffer;
        if (request.receive.address != 0u && request.receive.size != 0u && returnPointer != 0u)
        {
            const uint32_t physical = IopMemory::physicalAddress(returnPointer);
            if (physical < IopMemory::RamSize)
            {
                const uint32_t copySize = std::min<uint32_t>(request.receive.size, IopMemory::RamSize - physical);
                (void)m_host.writeGuest(request.receive.address, m_memory.ram().data() + physical, copySize);
                if (copySize < request.receive.size)
                    (void)m_host.zeroGuest(request.receive.address + copySize, request.receive.size - copySize);
            }
        }

        result.handled = true;
        result.resultAddress = request.receive.address;
        result.serverDispatchPolicy = ServerDispatchPolicy::Suppress;
        result.signalNowaitCompletion = true;
        result.signalCompletion = true;
        return result;
    }

    bool IopRpcBridge::handleSifCommand(uint32_t commandId, const void *packet, size_t packetSize, IopGuestExecutor &executor)
    {
        constexpr uint32_t kMaxPacketSize = 112u;
        const auto handlerIt = m_cmdHandlers.find(commandId);
        if (!packet || packetSize < 16u || packetSize > kMaxPacketSize ||
            handlerIt == m_cmdHandlers.end() || handlerIt->second.function == 0u)
            return false;

        if (m_cmdPacket == 0u)
            m_cmdPacket = m_memory.allocate(128u, 16u);
        if (m_cmdPacket == 0u || !m_memory.writeRam(m_cmdPacket, packet, packetSize))
            return false;

        // On hardware the handler runs in sifcmd's SIF interrupt: handler(packet, harg).
        const CmdHandler handler = handlerIt->second;
        if (std::getenv("VENDETTA_IOPLOG"))
        {
            std::ostringstream out;
            out << "[iop-sifcmd] EE cid=0x" << std::hex << commandId << " -> handler 0x" << handler.function
                << " packet at 0x" << m_cmdPacket;
            m_host.log(LogLevel::Info, out.str());
        }
        (void)executor.executeGuestFunction(handler.function, m_cmdPacket, handler.argument, 0u, 0u, handler.gp);
        return true;
    }

    void IopRpcBridge::onSifTransfer(const SifTransfer &transfer)
    {
        // The EE SIF transport owns the actual directional memory movement.
        // Services still receive both phases through IopSubsystem, but mirroring
        // IOP bytes through an equal-numbered EE address would alias two distinct
        // PS2 address spaces and can overwrite live game data.
        (void)transfer;
    }

    void IopRpcBridge::removeServersInRange(uint32_t base, uint32_t size)
    {
        for (auto server = m_servers.begin(); server != m_servers.end();)
        {
            const uint32_t function = IopMemory::physicalAddress(server->second.function);
            if (function >= base && function < base + size)
                server = m_servers.erase(server);
            else
                ++server;
        }
        for (auto handler = m_cmdHandlers.begin(); handler != m_cmdHandlers.end();)
        {
            const uint32_t function = IopMemory::physicalAddress(handler->second.function);
            if (function >= base && function < base + size)
                handler = m_cmdHandlers.erase(handler);
            else
                ++handler;
        }
    }

    bool IopRpcBridge::hasServer(uint32_t sid) const noexcept
    {
        const auto server = m_servers.find(sid);
        return server != m_servers.end() && server->second.function != 0u;
    }
}
