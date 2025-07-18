import numpy as np
import numbers


def format_aprepro_value(val):
    if isinstance(val, numbers.Real):
        val = np.double(val)
    elif isinstance(val, str):
        val = "\'" + val.strip('"').strip("'") + "\'"
    return val


def write_aprepro_file_from_dict(filename, params):
        with open(filename, "w") as fid:
            fid.write("{ECHO(OFF)}\n")
            for name, val in params.items():
                line_out = make_aprepro_string_from_name_val_pair(name, val)
                fid.write(line_out)
            fid.write("{ECHO(ON)}\n")


def make_aprepro_string_from_name_val_pair(name, val):
    val = format_aprepro_value(val)
    if isinstance(val, numbers.Real):
        line_out = f"# {name} = {{ {name} = {val:0.15E} }}\n"
    else:
        line_out = "# " + str(name) + " = { " + str(name) + " = " + str(val) + " }\n"
    return line_out


def parse_aprepro_variable_line(line):
    has_equals = "=" in line
    mod_line = line.strip().replace("#", '').replace("{", '').replace("}", "").replace(" ", "").split("=")
    try:
        mod_line[-1] = np.double(mod_line[-1])
    except ValueError:
        mod_line[-1] = mod_line[-1].strip('"')
    if has_equals:
        return mod_line[0], mod_line[-1]
    else:
        return None, None


                                           