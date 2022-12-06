import ray

ray.init(address="192.168.200.203:9394")

@ray.remote
def hello():
    return "Hello"

@ray.remote
def world():
    return "world!"

@ray.remote
def hello_world(a, b):
    return a + " " + b

a_id = hello.remote()
b_id = world.remote()
c_id = hello_world.remote(a_id, b_id)

hello = ray.get(c_id)

print(hello)
