import torchcomms
from torchcomms.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

comm = torchcomms.new_comm("ncclx", torch.device("cuda:0"), name="global")

mesh = init_device_mesh(
    mesh_dim_comms=(comm,),
    mesh_dim_names=("global",),
)
fully_shard(model, device_mesh=mesh)
