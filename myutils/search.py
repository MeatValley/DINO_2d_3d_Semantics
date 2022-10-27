

def search_xyz_in_file(x,y,z, path, eps=0.03):
    f = open(path)

    str_x = str(x)
    for line in f:
        
        if str_x in line:
            print(line)

search_xyz_in_file( x=2, y=3, z=4, path='office_28.txt')