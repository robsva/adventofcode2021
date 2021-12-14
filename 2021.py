import sys
import numpy as np

def import_data(file, ftype):
    with open(file, 'r') as fname:
        return [ftype(x.rstrip('\n')) for x in fname.readlines()]

### Day 1
def day1(file):
    data = import_data(file, int)
    increases = 0
    for i in range(len(data)-3):
        s1 = sum(data[i:i+3])
        s2 = sum(data[i+1:i+4])
        if s2 > s1:
            increases += 1
    print("Nr. of increases: " + str(increases))

### Day 2
def day2(file):
    data = import_data(file, str)
    aim = 0
    x = [0]
    y = [0]
    for action in data:
        a, b = action.split(' ')
        if a == 'forward':
            x.append(x[-1] + int(b))
            if aim == 0:
                y.append(y[-1])
            else:
                y.append(y[-1] + aim * int(b))
        elif a == 'up':
            aim -= int(b)
        elif a == 'down':
            aim += int(b)
    plt.plot(x, y)
    plt.xlabel('Forward position')
    plt.ylabel('Depth')
    plt.title('Final coordinate: (%d, %d)' % (x[-1], y[-1]))
    plt.show()

### Day 3
def day3(file):
    data = import_data(file, str)
    gamma = ''
    for i in range(len(data[0])):
        gamma += day3findcommon(data, i, 'direct')
    epsilon = ''.join(['1' if i == '0' else '0' for i in gamma])
    print('Gamma: ' + gamma + ' Epsilon: ' + epsilon)
    print('Power consumption: %d' % (int(gamma, 2) * int(epsilon, 2)))

    oxygen = data.copy()
    scrubber = data.copy()
    for i in range(len(data[0])):
        if len(oxygen) > 1:
            oxygen = day3findcommon(oxygen, i, 'most')
        if len(scrubber) > 1:
            scrubber = day3findcommon(scrubber, i, 'least')
    print('Oxygen: ' + oxygen[0] + ' Scrubber: ' + scrubber[0])
    print('Life support: %d' % (int(oxygen[0], 2) * int(scrubber[0], 2)))

def day3findcommon(lst, pos, type):
    s = 0
    for nr in lst:
        if int(nr[pos]):
            s += 1
    if s >= len(lst) / 2:
        if type == 'direct':
            return '1'
        elif type == 'most':
            return [x for x in lst if x[pos] == '1']
        elif type == 'least':
            return [x for x in lst if x[pos] == '0']
    else:
        if type == 'direct':
            return '0'
        elif type == 'most':
            return [x for x in lst if x[pos] == '0']
        elif type == 'least':
            return [x for x in lst if x[pos] == '1']

### Day 4
def day4(file):
    data = import_data(file, str)
    numbers = data[0].split(',')
    cards = []
    for i in range(0, int((len(data)-1)/6)):
        cards.append(day4card( data[2 + i*6: 2 + i*6 + 5] ))

    for number in numbers:
        for card in cards:
            card.bingo(float(number))

class day4card():
    def __init__(self, card):
        self.card = np.zeros((5, 5))
        for i, row in enumerate(card):
            self.card[i] = row.split()
        self.finished = False

    def bingo(self, number):
        if self.finished:
            return
        c = np.where(self.card == number)
        self.card[c] = None
        if any(np.sum(np.isnan(self.card), axis=0) == 5) or any(np.sum(np.isnan(self.card), axis=1) == 5):
            print('Bingo on number: %d' % number)
            print('Final score: %d' % (np.nansum(self.card) * number))
            self.finished = True

### Day 5
def day5(file):
    data = import_data(file, str)
    vents = [a.split(' -> ') for a in data]
    matrix = np.zeros((999, 999))
    for vent in vents:
        x1, y1 = map(int, vent[0].split(','))
        x2, y2 = map(int, vent[1].split(','))

        if x1 > x2:
            x = range(x1, x2 - 1, -1)
        else:
            x = range(x1, x2 + 1)
        if y1 > y2:
            y = range(y1, y2 - 1, -1)
        else:
            y = range(y1, y2 + 1)

        if x1 == x2:
            for i in y:
                matrix[i][x1] += 1
        elif y1 == y2:
            for i in x:
                matrix[y1][i] += 1
        else:
            for i, j in zip(y, x):
                matrix[i][j] += 1
    print('At least two overlaps: %d' % len(np.where(matrix > 1)[0]))

### Day 6
def day6(file):
    data = import_data(file, str)
    data = [int(x) for x in data[0].split(',')]
    daysLeft = np.zeros(10)
    for day in data:
        daysLeft[day] += 1
    for i in range(255):
        daysLeft = np.roll(daysLeft, -1)
        if daysLeft[0]:
            daysLeft[9] += daysLeft[0]
            daysLeft[7] += daysLeft[0]
            daysLeft[0] = 0
    print('Nr. of lanternfish after 256 days: %d' % np.sum(daysLeft))

### Day 7
def day7(file):
    data = import_data(file, str)
    data = [int(x) for x in data[0].split(',')]
    goal = np.median(data)
    fuel = 0
    for crab in data:
        fuel += abs(crab - goal)
    print('Fuel needed to linearly align the crabs to position %d: %d' % (goal, fuel))

    goal = int(np.floor(np.mean(data)))  # Example needs np.round...
    fuel = 0
    for crab in data:
        fuel += sum([1 * x for x in range(1, abs(crab - goal) + 1)])
    print('Fuel needed to cumulatively align the crabs to position %d: %d' % (goal, fuel)) 

### Day 8
def day8(file):
    data = import_data(file, str)
    inputData = []
    outputData = []
    for dat in data:
        dat = dat.split(' | ')
        inputData.append(dat[0].split(' '))
        outputData.append(dat[1].split(' '))

    count = 0
    sumVal = 0
    for inp, out in zip(inputData, outputData):
        for val in out:
            if len(val) in [2, 4, 3, 7]:
                count += 1

        # Identify first couple of digits
        digit = {'5': [], '6': []}
        solution = {}
        for val in inp:
            if len(val) == 2:
                digit['1'] = val
            elif len(val) == 4:
                digit['4'] = val
            elif len(val) == 3:
                digit['7'] = val
            elif len(val) == 7:
                digit['8'] = val
            elif len(val) == 5:  # 2, 3, or 5
                digit['5'].append(val)
            elif len(val) == 6:  # 0, 6, or 9
                digit['6'].append(val)

        # Identify first parts of the solution matrix
        solution['a'] = day8difference(digit['1'], digit['7'])
        solution['bd'] = day8difference(digit['1'], digit['4'])
        solution['cf'] = digit['1']
        solution['eg'] = day8difference(digit['8'], ''.join(solution.values()))

        # Sequentially find the remaining digits by process of elimination
        temp = day8difference(digit['8'], digit['6'][0])
        temp2 = day8difference(digit['8'], digit['6'][1])
        temp3 = day8difference(digit['8'], digit['6'][2])
        for val in [temp, temp2, temp3]:
            if val in solution['bd']:
                solution['d'] = val
                solution['b'] = day8difference(solution['bd'], val)
            elif val in solution['cf']:
                solution['c'] = val
                solution['f'] = day8difference(solution['cf'], val)
            elif val in solution['eg']:
                solution['e'] = val
                solution['g'] = day8difference(solution['eg'], val)
        digit['0'] = ''.join([solution[x] for x in ['a', 'b', 'c', 'e', 'f', 'g']])
        digit['2'] = ''.join([solution[x] for x in ['a', 'c', 'd', 'e', 'g']])
        digit['3'] = ''.join([solution[x] for x in ['a', 'c', 'd', 'f', 'g']])
        digit['5'] = ''.join([solution[x] for x in ['a', 'b', 'd', 'f', 'g']])
        digit['6'] = ''.join([solution[x] for x in ['a', 'b', 'd', 'e', 'f', 'g']])
        digit['9'] = ''.join([solution[x] for x in ['a', 'b', 'c', 'd', 'f', 'g']])

        v = ''
        for val in out:
            for dig, patt in digit.items():
                if not day8difference(val, patt):
                    v += dig
                    break
        sumVal += int(v)

    print('Nr. of times 1, 4, 7, or 8 appears: %d' % count)
    print('Sum of output values: %d' % sumVal)

def day8difference(s1, s2):
    return ''.join(set(s1).symmetric_difference(s2))

### Day 9
def day9(file):
    import matplotlib.pyplot as plt
    data = import_data(file, str)
    arr = np.ndarray((len(data), len(data[0]))).astype(int)
    for i, dat in enumerate(data):
        for j, v in enumerate(dat):
            arr[i][j] = v

    # Pad outside perimiter with the global maxima
    m = np.max(arr)
    arr = np.pad(arr, (1, 1), mode='constant', constant_values=(m, m))
    loc_min = []
    for x in range(arr.shape[0] - 1):
        for y in range(arr.shape[1] - 1):
            if (arr[x, y] < arr[x, y+1] and 
                arr[x, y] < arr[x, y-1] and 
                arr[x, y] < arr[x+1, y] and 
                arr[x, y] < arr[x-1, y]):
                loc_min.append((x, y))
    s = sum([arr[x] for x in loc_min]) + len(loc_min)
    print('Sum of risk levels: %d' % s)

    # Find sizes of the local minima basins
    basins = []
    full_map = np.zeros(arr.shape)
    for loc in loc_min:
        bool_arr = np.zeros(arr.shape)
        bool_arr = day9basin(arr, *loc, bool_arr)
        basins.append(np.sum(bool_arr))
        full_map += bool_arr
    print('Cumulative product of the three largest basins: %d' % np.cumprod(sorted(basins)[-3:])[-1])

    fig = plt.figure()
    plt.imshow(full_map)
    fig = plt.figure()
    plt.imshow(arr)
    plt.show()

def day9basin(arr, x, y, bool_arr):
    if arr[x, y] == 9:
        return bool_arr
    else:
        bool_arr[x, y] = 1
    if arr[x, y] < arr[x, y+1]:
        bool_arr = day9basin(arr, x, y+1, bool_arr)
    if arr[x, y] < arr[x, y-1]:
        bool_arr = day9basin(arr, x, y-1, bool_arr)
    if arr[x, y] < arr[x+1, y]:
        bool_arr = day9basin(arr, x+1, y, bool_arr)
    if arr[x, y] < arr[x-1, y]:
        bool_arr = day9basin(arr, x-1, y, bool_arr)
    return bool_arr

### Day 10
def day10(file):
    data = import_data(file, str)

    # Find corrupted and incomplete lines
    corrupted = []
    incomplete = []
    for line in data:
        line = list(line)
        print(line)
        inc = ''
        while len(line) > 0:
            if len(line) == 1:
                corrupted.append(line)
                break
            line, inc = day10close(line, 0, line[0], inc)
            if inc:
                incomplete.append(inc)

    points = {')': 3, ']': 57, '}': 1197, '>': 25137}
    s = sum([points[x] for x in corrupted])
    print('Total syntax error score: %d' % s)

    points = {')': 1, ']': 2, '}': 3, '>': 4}
    s = []
    for chars in incomplete:
        v = 0
        for char in chars:
            v *= 5
            v += points[char]
        s.append(v)
    print('Autocomplete middle score: %d' % np.median(sorted(s)))

def day10close(arr, index, char, inc):
    mtch = {'(': ')', '[': ']', '{': '}', '<': '>'}
    corr = {'(': ']}>', '[': ')}>', '{': ')]>', '<': ')]}'}

    if len(arr) < index+2:
        print('Incomplete sequence, missing ', mtch[char])
        inc += mtch[char]
        return '', inc

    # Keep looking
    if arr[index+1] in '([{<':
        print('Continue with: ', arr[index+1])
        arr, inc = day10close(arr, index+1, arr[index+1], inc)

    if len(arr) == 1:
        return arr, inc
    elif len(arr) < index+2:
        print('Incomplete sequence, missing ', mtch[char])
        inc += mtch[char]
        return '', inc

    # Correct closing bracket, removing from list
    if arr[index+1] == mtch[char]:
        print('Correct: ', char, arr[index+1])
        arr.pop(index+1)
        arr.pop(index)
        return arr, inc

    # Corrupted closing bracket
    elif arr[index+1] in corr[char]:
        print('Corrupted: ', char, arr[index+1])
        return arr[index+1], inc

    return day10close(arr, index, arr[index], inc)

### Day 11
def day11(file):
    data = import_data(file, str)
    arr = np.ndarray((len(data), len(data[0]))).astype(int)
    for i, dat in enumerate(data):
        for j, v in enumerate(dat):
            arr[i, j] = v
    ones = np.ones(arr.shape).astype(int)
    blink = np.ones((3, 3)).astype(int)
    flashes = 0

    for i in range(1000):
        arr += ones
        blinked = np.zeros(arr.shape).astype(bool)
        while (arr > 9).any():
            x, y = np.where(arr > 9)
            for xi, yi in zip(x, y):
                if not blinked[xi, yi]:
                    blinked[xi, yi] = True
                    arr = day11blink(arr, xi, yi, blink)
                    arr[xi, yi] = 0
        arr[blinked] = 0
        flashes += np.sum(blinked)
        if i == 99:
            print('Total flashes after 100 steps: %d' % flashes)
        if blinked.all():
            print('All octopuses flash on step: %d' % (int(i)+1))
            break

def day11blink(arr, x, y, blink):
    arr = np.pad(arr, 1)
    arr[x:x+blink.shape[0], y:y+blink.shape[1]] += blink
    return arr[1:-1, 1:-1]

### Day 12
def day12(file):
    from collections import defaultdict
    data = import_data(file, str)
    paths = defaultdict(list)
    for dat in data:
        a, b = dat.split('-')
        if a == 'start':
            paths[a].append(b)
        elif b == 'start':
            paths[b].append(a)
        else:
            paths[a].append(b)
            paths[b].append(a)
    npaths = []
    for room in paths['start']:
        seq = ['start']
        _, npaths = day12step(paths, room, npaths, seq)
    print('Paths through the cave system: %d' % len(npaths))

def day12step(paths, room, npaths, seq):
    # Evaluate current room
    if room == 'end':
        seq.append(room)
        npaths.append(seq.copy())
        seq.pop(-1)
        return seq, npaths
    elif room.islower() and room in seq:
        if any(i > 1 for i in [seq.count(x) for x in seq if x.islower()]):
            return seq, npaths
    seq.append(room)

    # Iterate through adjacent rooms
    for adjacent in paths[room]:
        seq, npaths = day12step(paths, adjacent, npaths, seq)

    # Remove last room and return the rest of the sequence
    seq.pop(-1)
    return seq, npaths

### Day 13
def day13(file):
    data = import_data(file, str)
    points = []
    instr = []
    for dat in data:
        if ',' in dat:
            y, x = map(int, dat.split(','))
            points.append((x, y))
        elif dat == '':
            continue
        else:
            instr.append(dat.split(' ')[2])
    x = max(points, key = lambda i: i[0])[0] + 1
    y = max(points, key = lambda i: i[1])[1] + 1
    arr = np.zeros((x, y)).astype(bool)
    for p in points:
        arr[p] = 1

    for i, inst in enumerate(instr):
        line, row = inst.split('=')
        row = int(row)
        if line == 'y':
            arr1 = arr[:row, :]
            arr2 = np.flipud(arr[row+1:, :])
        elif line == 'x':
            arr1 = arr[:, :row]
            arr2 = np.fliplr(arr[:, row+1:])
        arr = arr1 + arr2

        if i == 0:
            print('Nr. of points after first fold: %d' % np.sum(arr))

    import matplotlib.pyplot as plt
    plt.imshow(arr)
    plt.title('Thermal camera code:')
    plt.show()

### Day 14
def day14(file):
    from collections import defaultdict
    data = import_data(file, str)
    start = data[0]
    rules = {}
    for dat in data[2:]:
        a, b = dat.split(' -> ')
        rules[a] = b

    # Start with initial set of combinations
    formula = defaultdict(int)
    for i in range(len(start)-1):
        formula[start[i] + start[i+1]] = 1

    for i in range(40):
        tmp = defaultdict(int)
        for key, val in formula.items():
            inp = rules[key]
            tmp[key[0] + inp] += val
            tmp[inp + key[1]] += val
            formula[key] -= val
        # Remove combinations that has been exhausted
        vals = formula.copy()
        for key, val in vals.items():
            if val <= 0:
                formula.pop(key)
        # Populate with new combinations
        for key, val in tmp.items():
            formula[key] += val

    occurence = defaultdict(int)
    for key, val in formula.items():
        occurence[key[1]] += val
    occurence[start[0]] += 1
    vals = occurence.values()
    print('Most common element minus least common element: %d' % (max(vals) - min(vals)))

#14.txt
#input/14.txt
day = sys.argv[1].lstrip('input/').split('.')[0].rstrip('ex')
locals()["day" + day](sys.argv[1])
