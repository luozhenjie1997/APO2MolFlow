def clean_sdffile(filename, is_str=False):
    # 将元素名称的第二个字母转换为小写（例如 FE→Fe），以便 openbabel 能正确解析它。
    lines2 = []
    if is_str:
        lines = filename
    else:
        with open(filename) as f:
            lines = f.readlines()
    num_atoms = int(lines[3][:3])
    for i in range(len(lines)):
        if i>=4 and i<4+num_atoms:
            lines2.append(lines[i][:32]+lines[i][32].lower()+lines[i][33:])
        else:
            lines2.append(lines[i])
    molstring = '\n'.join(lines2) if is_str else ''.join(lines2)

    return molstring
