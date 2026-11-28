import re, sys
src = open(sys.argv[1], encoding='utf-8').read()
src2 = re.sub(r'//[^\n]*', '', src)
src2 = re.sub(r'/\*.*?\*/', '', src2, flags=re.S)
src2 = re.sub(r'"(?:[^"\\]|\\.)*"', '""', src2)
src2 = re.sub(r"'(?:[^'\\]|\\.)'", "'x'", src2)
print('braces:', src2.count('{') - src2.count('}'),
      'parens:', src2.count('(') - src2.count(')'),
      'brackets:', src2.count('[') - src2.count(']'))
