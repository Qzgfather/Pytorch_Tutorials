f = open('./data.txt', 'r', encoding='utf-8').readlines()
cheat_history = open('ff.txt', 'a', encoding='utf-8')
num = 0
for i in range(len(f)):
    if "我欲乘风归去" in f[i] or "蠢菲" in f[i]:
        print(f[i+1], end='')
        cheat_history.write(f[i+1])
        num += 1

print('\n', num)
cheat_history.close()
