

import javalang
lines = ""
dict = {}
idx = 4

numeric_idx = 1
string_idx = 2
comment_idx = 3
with open('Solution.java') as f:
    lines = f.read()


tokens = list(javalang.tokenizer.tokenize(lines))


for token in tokens:
    # case numerical value
    if token.value.isnumeric():
        dict[token.value] = numeric_idx
    # case user defined string
    elif token.value[0] == '\"':
        dict[token.value] = string_idx
    # case comment
    elif (token.value[0:3] == "/**"):
        dict[token.value] = comment_idx

    else:
        dict[token.value] = idx
        idx += 1

print(dict)
