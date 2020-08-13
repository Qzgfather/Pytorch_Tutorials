"""
0 身体不舒服
1   游戏
2   哥哥
3   生气 7   不高兴
4   睡觉
5   学习
6   焦虑
10  道歉
8   疑问
9   时政

"""
# f = open('ff.txt', 'r', encoding='utf-8').readlines()
#
# for i in f:
#     f_data = open('ff2.txt', 'a', encoding='utf-8')
#     if '图片' not in i:
#         print(i, end='')
#         number = input("请输入标签：")
#         data = i.strip('\n')
#         input_data = data + '\t' + number + '\n'
#         f_data.write(input_data)
#         f_data.close()
f_data = open('ff2.txt', 'r', encoding='utf-8').readlines()
f_data2 = open('train.txt', 'a', encoding='utf-8')
for i in f_data:
    if i.split('\t')[1] == '\n':
        pass
    else:
        f_data2.write(i)

f_data2.close()
