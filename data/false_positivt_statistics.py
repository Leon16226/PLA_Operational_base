# 统计误报数据

File_Root = '/Users/apple/Desktop/1月误报结果/clk.txt'
labels = {'Bag': [], 'Bottle': [], 'Box': [],
          'MealBox': [], 'BullBarrels': []}
thresh = 0.6

if __name__ == "__main__":
    with open(File_Root, 'r') as f:
        print("开始读取文件")
        one_item_array = {}
        for line in f.readlines():
            print(line)
            line = line.strip('\n')

            split_narray = line.split()
            # 测试
            # print("***" in split_narray[0])

            if split_narray[0] in list(labels.keys()):
                if split_narray[0] in list(one_item_array.keys()):
                    temp = float(one_item_array[split_narray[0]])
                    one_item_array[split_narray[0]] = float(split_narray[2]) if temp < float(split_narray[2]) else temp
                    print(split_narray[2], len(one_item_array))
                else:
                    one_item_array[split_narray[0]] = float(split_narray[2])
                    print(split_narray[2], len(one_item_array))
            elif ( ("***" in split_narray[0]) & (len(one_item_array) > 0)):
                # 取最大的值
                values = list(one_item_array.values())
                print("最大值",values)
                max_value = max(values)

                for key, vaule in one_item_array.items():
                    if max_value == vaule:
                        temp = labels[key]
                        temp.append(vaule)
                        labels[key] = temp
                one_item_array.clear()
    f.close()

    print(labels)

    # 误报总数
    # all = list()
    # all.extend(list(a) for a in labels.values())
    # all_size = 0
    # all_size = len(all)
    # print(all)
    # print("总数是%i"%all_size)

    all_size = 0
    for a in list(labels.values()):
        print(a)
        all_size += len(a)
    print("总数是%i" % all_size)

    # 误报类别，分别多少个，百分比
    false_kinds = {}
    print('类别 小于%.1f 大于%.1f 总数 百分比' % (thresh, thresh))
    for key, value in labels.items():
        false, postive = 0, 0
        for a in value:
            if float(a) >= thresh:
                postive += 1
            else:
                false += 1
        all = false + postive
        false_kinds[key] = [int(false), int(postive), int(all), all / all_size]
        print('%s %i %i %i %.5f' % (key, int(false), int(postive), int(all), all / all_size))
