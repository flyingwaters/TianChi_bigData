from Model import*
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import*
from tensorboardX import SummaryWriter
from preprocessing_validition import*
import csv
batch_size = 256
hidden_size = 50

writer = SummaryWriter()

train_d = poetrySet(train_x, train_y)
test_d = poetrySet(test_x, test_y)
train_loader = DataLoader(dataset=train_d, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_d, batch_size=batch_size, shuffle=True, drop_last=True)
net = Gru()
# 初始化
net = net.cuda()
criterion = nn.MSELoss(reduce=True, size_average=True)

# size_average = True 求batch中的平均loss
# reduce = True 返回标量的loss，返回向量的loss
optimizer = optim.Adam(net.parameters(), lr=0.000001)


def test(model, epoch):
    # global best_acc
    model.eval()
    test_sum_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.float().cuda(), targets.float().cuda()
            outputs = model(inputs)
            test_loss = criterion(outputs, targets)
            test_sum_loss += test_loss

        print("batch_num:", batch_idx+1)
        print('test_loss: {}  epoch:{}'.format(test_sum_loss/(batch_idx+1), epoch))
        writer.add_scalar('runs/scalars', test_sum_loss.cpu()/(batch_idx+1), epoch)
        return test_sum_loss.cpu()/(batch_idx+1)

def train(model, epoch):
    # colony
    print('\n epoch: %d' % epoch)
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.float().cuda(), targets.float().cuda()
        outputs = model(inputs)
        # regression 的值, one-hot编码
        loss = criterion(outputs, targets)
        # batch_first
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('train_loss:%0.3f ' % (train_loss/(batch_idx + 1)))
    writer.add_scalar('runs/scalars', train_loss/(batch_idx + 1), epoch)
    test_s_loss = test(net, epoch)

    train_a_loss = train_loss.cpu() / (batch_idx+1)
    if test_s_loss < 0.70 and (train_loss.cpu()/(batch_idx+1)) < 0.70:
        torch.save(net.state_dict(), 'model/net_{}_train_loss:{.3f}_test_loss:{.3f}_parameters.pkl'.format(epoch,
                                                                                                     train_a_loss,
                                                                                                     float(test_s_loss.cpu())))
    if epoch == 99999:
        torch.save(net.state_dict(), 'model/net_{}_train_loss:{.3f}_test_loss:{.3f}_parameters.pkl'.format(epoch,
                                                                                                     train_a_loss,
                                                                                                     float(test_s_loss.cpu())))

def val_out(model):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(fin_validation, dtype=torch.float32))
        headers = ["id", "digging_for_happiness"]
        values = []
        for i_index, j_s in enumerate(iter(out)):
            values.append({"id": "{}".format(i_index+8001), "digging_for_happiness": "{}".format(j_s.item())})

    with open("data/res.csv", "w", encoding="gb18030") as f:
        write_csv = csv.DictWriter(f, headers)
        write_csv.writeheader()
        write_csv.writerows(values)
    print("over!!!!!!!!!!!!!!!!")

ik = int(input("please input? train: 0 val： 1.\n"))
if ik == 0:
    for i in range(100000):
        train(net, i)
else:
    net_q = Gru().float()
    net_q.load_state_dict(torch.load("model/net_11667_train_loss:0.6798129677772522_test_loss:0.6974645853042603_parameters.pkl"))
    val_out(net_q)
