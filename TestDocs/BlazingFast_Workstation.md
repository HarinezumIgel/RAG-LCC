# BlazingFast Workstation Technical Specifications

The BlazingFast Workstation is an enterprise-grade computing system engineered specifically for data science, artificial intelligence (AI) model training, advanced scientific simulations, and hybrid vector retrieval workloads.

## Core Compute Architecture

- **Processors**: Dual Enterprise Platinum 8500-series processors. Providing up to 120 physical cores (240 threads) per system. Base clock speed is 2.8 GHz, with single-core turbo frequencies up to 4.6 GHz.
- **Memory**: Supports up to 2 TB of Error-Correcting Code (ECC) DDR5 RAM running at 4800 MHz across 16 DIMM slots. The high-bandwidth memory architecture is optimized for loading large vector databases and LLM context windows entirely into RAM.

## Storage and I/O

- **Storage Subsystem**: 8x 4TB NVMe U.2 PCIe Gen 5 SSDs, configured by default in RAID 10 for a total of 16TB of redundant, ultra-fast local storage. Delivers sequential read speeds up to 28,000 MB/s and extremely low latency for database lookups.
- **Networking**: Integrated dual 100GbE SFP28 network interfaces for high-throughput cluster environments, alongside an independent out-of-band management interface (IPMI).
- **Expansion**: 6x PCIe Gen 5 x16 slots, providing sufficient lanes for multiple accelerator cards, dedicated hardware RAID controllers, or additional network interface cards (NICs).

## Hardware Acceleration and AI

- **Graphics**: Accommodates up to Quad-Datacenter RTX 6000 Ada Generation GPUs. With 48GB VRAM per GPU (192GB total), the system can easily support distributed inference, fine-tuning of 70B+ parameter language models, and hardware-accelerated document classification.
- **Tensor Cores**: Embedded fourth-generation Tensor Cores drastically reduce processing times for matrix multiplication, a core requirement for embedding generation.

## Power and Thermal Management

- **Power Delivery**: Dual 2000W redundant Titanium-rated power supplies (N+1 configuration). Requires a 200V-240V circuit due to the massive power draw at full compute load.
- **Thermals**: A specialized closed-loop liquid cooling system ensures both GPUs and CPUs remain below 70°C even under sustained 24/7 workloads, preventing thermal throttling.

## Software Ecosystem

- **OS Support**: Certified for Ubuntu 22.04 LTS and Windows 11 Pro for Workstations.
- **Pre-installed Stack**: Ships optionally with a fully configured AI stack, including CUDA 12, PyTorch, Docker, and pre-configured Python environments tuned for RAG (Retrieval-Augmented Generation) pipeline development.
