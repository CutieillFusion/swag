import random
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

# Helper: call remote methods on an RRef's local value.
def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

####################################
# Remote embedding server on CPU.
####################################
class RemoteEmbeddingServer(torch.nn.Module):
    def __init__(self):
        super(RemoteEmbeddingServer, self).__init__()
        # Create the embedding on CPU.
        self.embedding = torch.nn.EmbeddingBag(NUM_EMBEDDINGS, EMBEDDING_DIM, mode="sum")
        self.optimizer = optim.SGD(self.embedding.parameters(), lr=0.05)
    
    def forward(self, indices, offsets):
        # Work with CPU tensors.
        return self.embedding(indices, offsets)

    def update(self, indices, offsets, grad_output):
        self.optimizer.zero_grad()
        out = self.embedding(indices, offsets)
        out.backward(grad_output)
        self.optimizer.step()
        return True

####################################
# Local (Dense) Model for trainers.
####################################
class HybridLocalModel(torch.nn.Module):
    def __init__(self, device):
        super(HybridLocalModel, self).__init__()
        self.fc = torch.nn.Linear(EMBEDDING_DIM, 8).to(device)
        self.device = device

    def forward(self, emb_out):
        return self.fc(emb_out.to(self.device))

####################################
# Trainer routine.
####################################
def run_trainer(remote_emb_rref, trainer_group_rank, num_trainers):
    backend = "nccl" if torch.cuda.is_available() else "mpi"
    print(f"Trainer group rank {trainer_group_rank}: initializing process group with backend {backend}")
    dist.init_process_group(
        backend=backend,
        rank=trainer_group_rank,
        world_size=num_trainers,
        init_method="tcp://localhost:29500",
    )

    device = torch.device("cuda", trainer_group_rank) if torch.cuda.is_available() else torch.device("cpu")
    local_model = HybridLocalModel(device)
    ddp_model = DDP(local_model, device_ids=[trainer_group_rank] if torch.cuda.is_available() else None)
    local_optimizer = optim.SGD(ddp_model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    try:
        for epoch in range(25):
            for _ in range(10):
                num_indices = random.randint(20, 50)
                indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)
                offsets = []
                start = 0
                while start < num_indices:
                    offsets.append(start)
                    start += random.randint(1, 10)
                offsets = torch.LongTensor(offsets)
                
                # Ensure indices and offsets are CPU tensors.
                emb_out = rpc.rpc_sync(
                    remote_emb_rref.owner(),
                    _call_method,
                    args=(RemoteEmbeddingServer.forward, remote_emb_rref, indices.cpu(), offsets.cpu())
                )
                # Now move the result to the desired device.
                emb_out = emb_out.to(device)
                
                output = ddp_model(emb_out)
                target = torch.LongTensor([random.randint(0, 7) for _ in range(output.size(0))]).to(device)
                loss = criterion(output, target)
                
                local_optimizer.zero_grad()
                loss.backward()
                local_optimizer.step()

                # Simulate remote update with dummy gradient.
                dummy_grad = torch.ones_like(emb_out.cpu())
                rpc.rpc_sync(
                    remote_emb_rref.owner(),
                    _call_method,
                    args=(RemoteEmbeddingServer.update, remote_emb_rref, indices.cpu(), offsets.cpu(), dummy_grad)
                )
            print(f"Trainer {trainer_group_rank}: Completed epoch {epoch + 1}")
    finally:
        dist.destroy_process_group()

####################################
# Master and Parameter Server roles.
####################################
def run_master_and_ps(rank, world_size):
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29501")
    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # Create remote embedding server on CPU.
        remote_emb_rref = rpc.remote("ps", RemoteEmbeddingServer)
        num_trainers = world_size - 2
        futs = []
        for global_trainer_rank in range(2, world_size):
            trainer_name = f"trainer{global_trainer_rank}"
            trainer_group_rank = global_trainer_rank - 2
            fut = rpc.rpc_async(
                trainer_name,
                run_trainer,
                args=(remote_emb_rref, trainer_group_rank, num_trainers)
            )
            futs.append(fut)
        for fut in futs:
            fut.wait()
    elif rank == 1:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
    rpc.shutdown()

def run_worker(global_rank, world_size):
    if global_rank in (0, 1):
        run_master_and_ps(global_rank, world_size)
    else:
        trainer_name = f"trainer{global_rank}"
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method="tcp://localhost:29501")
        rpc.init_rpc(
            trainer_name,
            rank=global_rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        rpc.shutdown()

if __name__ == "__main__":
    world_size = 4  # 1 master, 1 ps, and at least 1 trainer.
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
