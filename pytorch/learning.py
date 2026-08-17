import torch
print(torch.__version__)

if torch.cuda.is_available():
    print("GPU is available")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print("GPU is not available. Using CPU instead.")




#Creating a tensor ----   There are bunch of ways to create a tensor in pytorch.



x1 = torch.empty(3, 4)                       #using empty() function
print('x1:', x1)
print('Type of x1:', type(x1))



'''
If we create a tensor using the empty() function, it will create a tensor with
uninitialized values, as they will create tensors with known values. Here the values are
random and they are some gerbage values from the memory. So, we should not use the empty() function
to create a tensor if we want to have known values. Instead we can use the ones() or zeros() function
to create a tensor with known values. The ones() function will create a tensor with all values as 1 
and the zeros() function will create a tensor with all values as 0. The rand() function will create a
tensor with random values between 0 and 1. The rand() function will create a tensor with random values
hence if we run the code multiple times, we will get different values each time.
'''


x2 = torch.ones(3, 4)                        #using ones() function
print('x2:', x2)

x3 = torch.zeros(2, 3)                       #using zeros() function
print('x3:', x3)



x4 = torch.rand(3, 4)                        #using rand() function
print('x4:', x4)

x5 = torch.rand(3, 4)                       #using  rand() function
print('x5:', x5)

'''
We run the code torch.rand() two times and we got different values each time. This is because the
rand() function generates random values between 0 and 1. If we want to have the same random values
each time we run the code, we can set the seed the same value using the torch.manual_seed() function.
'''

torch.manual_seed(100)  # Setting the seed for reproducibility
x6 = torch.rand(3, 4)  # This will now produce the same values each time
print('x6:', x6)

torch.manual_seed(100)  # Setting the seed again for reproducibility
x7 = torch.rand(3, 4)  # This will produce the same values as x6
print('x7:', x7)




# Using tensor

x8 = torch.tensor([[1,2,3],[4,5,6]])
print('x8:', x8)



# Other ways to create a tensor

print("using arange ->", torch.arange(0,10,2))


print("using linspace ->", torch.linspace(0,20,10)) 

print("using eye ->", torch.eye(5))      # creating a 5x5 identity matrix ; eye -> identity matrix , 5 -> order 5 i.e. 5x5

print("using full ->", torch.full((3,3),5))   # creating a 3x3 matrix whose all entries are 5




# Tensor Shapes

x9 = torch.tensor([[1,2,3],[4,5,6]])
print(x9.shape)

