from matcal.core.object_factory import IdentifierByTestFunction


def plain_text_dictionary_report(filename:str, report_dict:dict)->None:
    with open(filename, 'w') as f:
        lines = ""
        for key, value in report_dict.items():
            lines += f"{key}={value}\n"
        f.write(lines)            


MatCalParameterReporterIdentifier = \
    IdentifierByTestFunction(plain_text_dictionary_report)