import math

import  torch

def main():

  a = torch.rand(25, 1,32,32)


  b=a.clone()

  dim0, dim1, dim2, dim3 = a.shape
  print(dim2)
  print(dim3)
  print(a.max())
  print(a.min())

  # # 遍历张量
  # for k in range(dim0):
  #     for i in range(dim2):
  #         for j in range(dim3):
  #             element = a[k][0][i][j].item()
  #             print(element)
  #             if element<-math.pi:
  #                 a[k][0][i][j]=element+2*math.pi
  #
  #             if element>math.pi:
  #                 a[k][0][i][j] = element - 2 * math.pi


  # flag=a.equal(b)
  print(3.1600e+01)








if __name__ == '__main__':
    main()
