#交叉熵
class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        # 他说已经弃用
        self.loss = torch.nn.NLLLoss(weight)
        # self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        # return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

#差分部分，使用图片平移像素代替求导
class residue(torch.nn.Module):
    def __init__(self):
        super(residue, self).__init__()

    def forward(inputs, unwarpped):
        w_pad_x = nn.ZeroPad2d(padding=(0, 1, 0, 0))
        w_pad_y = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        u_pad_x = nn.ZeroPad2d(padding=(0, 1, 0, 0))
        u_pad_y = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        w_grad_img_x = torch.mean(inputs[:, :, :, :-1] - inputs[:, :, :, 1:], 1, keepdim=True)

        w_grad_img_y = torch.mean(inputs[:, :, :-1, :] - inputs[:, :, 1:, :], 1, keepdim=True)
        un_grad_img_x = torch.mean(unwarpped[:, :, :, :-1] - unwarpped[:, :, :, 1:], 1, keepdim=True)
        un_grad_img_y = torch.mean(unwarpped[:, :, :-1, :] - unwarpped[:, :, 1:, :], 1, keepdim=True)
        w_grad_img_x = w_pad_x(w_grad_img_x)
        w_grad_img_y = w_pad_y(w_grad_img_y)
        un_grad_img_x = u_pad_x(un_grad_img_x)
        un_grad_img_y = u_pad_y(un_grad_img_y)
        resu = torch.mean(torch.abs(w_grad_img_x - un_grad_img_x) + torch.abs(w_grad_img_y - un_grad_img_y))

        return resu
    #l1
metric_loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

#使用损失函数
if args.cuda:
    images = images.cuda()
    labels = labels.cuda()

inputs = Variable(images)
targets = Variable(labels)
outputs = model(inputs, only_encode=enc)
#级次图
order1 = outputs.max(1)[1].unsqueeze(1).data
#erfnet的encoder部分有下采样
if enc == True:
    inputs = Resize(32, Image.NEAREST)(inputs)
optimizer.zero_grad()
#差分
loss1 = residue.forward(inputs, order1 * math.pi * 2 + inputs)
#交叉熵
loss2 = CrossEntropyLoss2d(outputs, targets[:, 0])
loss3 = metric_loss(order1 * math.pi * 2 + inputs, inputs + targets * 2 * math.pi)
loss = loss1 + loss3 + loss2

loss.backward()
optimizer.step()