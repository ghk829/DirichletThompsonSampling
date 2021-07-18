import matplotlib.pyplot as plt
import numpy as np
contents_ctr = np.array([0.1, 0.3, 0.3,0.1, ])
cans_str = [np.array([0.55, 0.9, 0.7]), np.array([0.8, 0.3, 0.4, 0.1])]
length = len(contents_ctr)
for x in cans_str:
    length += len(x)
prior = np.ones_like(contents_ctr)
candidates = list(range(len(contents_ctr)))
norm_contents_ctr = contents_ctr/ sum(contents_ctr)
print('각 컨텐츠에 따른 CTR')
for x, y in zip(candidates,contents_ctr):
    print(x,'번쨰 아이템 CTR : ',y)
selected = np.zeros_like(prior)

grid_size = 3
fig,ax=plt.subplots(grid_size,grid_size)
x = 0

decays = {x:1 for x in range(length)}
cluster_decay = [len(contents_ctr)]
from itertools import cycle
circle = cycle(range(grid_size))
num_iter = 100000
plt.suptitle('selected 1,2,3th rank')
ploted = 0
for i in range(num_iter):
    _prior = np.copy(prior)
    cluster_decay_factor = pow(10,len(cluster_decay))
    _cluster_decay = cluster_decay.copy()
    for k,l in decays.items():
        if len(_prior) == k:
            break
        if k >= _cluster_decay[0]:
            cluster_decay_factor/=10
            _cluster_decay.pop(0)
        _prior[k]/=(l*cluster_decay_factor)

    # for k,l in decays.items():
    #     if len(_prior) == k:
    #         break
    #     _prior[k]/=(l)
    # print(decays)
    # print(_prior)
    # print(prior)

    recommend = np.random.dirichlet(_prior,1)
    res = recommend[0].argsort()[::-1]

    click = np.random.choice(candidates, p=norm_contents_ctr)
    selected[res[0]] +=1
    selected[res[1]] += 1
    selected[res[2]] += 1
    prior[click] +=1

    for j in range(len(contents_ctr)):
        decays[j]+=1
    sec = int(num_iter/grid_size)

    if i % int(sec/grid_size)==0:
        y = next(circle)

    if i % int(sec/grid_size)==0:
        ax[x,y].bar(candidates,selected)
        # print(decays)
        # print()
        # print(prior)
        # print()
        # print(_prior)
        # print('################')
        if x > 0:
            ax[x, y].set_title(f"{x+1}'th cans added {i}'th iter")
        else:
            ax[x,y].set_title(f"{i}'th iter")
        ploted+=1
        if ploted * (x+1) == grid_size**2:
            break
        if ploted == grid_size:
            x +=1
            ploted =0
            cans = cans_str.pop(0)
            contents_ctr = np.concatenate([contents_ctr, cans])
            prior = np.concatenate([prior, np.ones_like(cans)])
            candidates = list(range(len(contents_ctr)))
            norm_contents_ctr = contents_ctr / sum(contents_ctr)
            selected = np.concatenate([selected, np.zeros_like(cans)])
            cluster_decay.append(len(contents_ctr))
            # print(decays)
            print('각 컨텐츠에 따른 CTR')
            for z, n in zip(candidates, contents_ctr):
                print(z, '번쨰 아이템 CTR : ', n)

print(norm_contents_ctr)
print(selected/ sum(selected))
plt.show()