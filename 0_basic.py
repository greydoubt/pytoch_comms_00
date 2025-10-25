import torchcomms

# Eagerly initialize a communicator using MASTER_PORT/MASTER_ADDR/RANK/WORLD_SIZE environment variables 
provided by torchrun.
# This communicator is bound to a single device.
comm = torchcomms.new_comm("ncclx", torch.device("cuda"), name="my_comm")
print(f"I am rank {comm.get_rank()} of {comm.get_size()}!")

t = torch.full((10, 20), value=comm.rank, dtype=torch.float)

# run an all_reduce on the current stream
comm.allreduce(t, torchcomms.ReduceOp.SUM, async_op=False)

# run an all_reduce on the background stream 
work = comm.allreduce(t, torchcomms.ReduceOp.SUM, async_op=True)
work.wait()

# split a communicator into groups of 8
split_groups = torch.arange(comm.get_size()).view(-1, 8).tolist()
tp_comm = comm.split(split_groups)

