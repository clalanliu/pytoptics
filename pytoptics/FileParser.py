def convert_list_of_string_to_float(str_list):
    return [float(i) for i in str_list]


def CodeVSurfaceParser(surface_str):
    substrs = surface_str.split()
    R = None
    Thickness = None
    Material = 'AIR'
    R = float(substrs[1])
    if R==0:
        R = 9999999999
    Thickness = float(substrs[2])
    if len(substrs) > 3:
        Material = substrs[3].split('_')[0]
   
    return R, Thickness, Material
        
