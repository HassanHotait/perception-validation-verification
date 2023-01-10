import torch

# t1=torch.abs(torch.tensor([1.1, 2.4, 3.2]))
# # t2=torch.abs(torch.tensor([-10, -22, 3]))

# # print(t1/((t2*2)+1))

# # print(type(t1))
# # print(t1)
# # print(t1.type(torch.int64))

# for i,element in enumerate(t1):
#     print(element.item())
#     t1[i]=4
#     print(element.item())

# print(t1)

Z=torch.empty(50)
print(Z)

for i,element in enumerate(Z):
    Z[i]=4

print(Z)