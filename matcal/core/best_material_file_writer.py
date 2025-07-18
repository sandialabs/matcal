from datetime import date

from matcal.core.object_factory import DefaultObjectFactory, ObjectCreator
from matcal.core.utilities import get_username_from_environment


class DefaultResultsFileWriter:
    def __init__(self, parameters):
        self.parameters = parameters

    def write(self, filename):
        line_list = self._create_lines()
        with open(filename, 'w') as results_file:
            for line in line_list:
                results_file.write(line)

    def _create_lines(self):
        line_list = self._get_header_lines()
        line_list += self._get_parameter_lines()
        line_list.append('###################################\n')
        return line_list

    def _get_header_lines(self):
        today = date.today()
        header_lines = []
        header_lines.append('###################################\n')
        header_lines.append(f'# Calibrated by: {get_username_from_environment()}\n')
        header_lines.append("# Calibration Finish Date:\n")
        header_lines.append(f"# Day: {today.day} Month: {today.month} Year: {today.year}\n")
        return header_lines

    def _get_parameter_lines(self):
        lines = []
        for name, value in self.parameters.items():
            lines.append("{} = {}\n".format(name, value))
        return lines


class DefaultResultsFileWriterCreator(ObjectCreator):
    
  def __call__(self, results):
      return DefaultResultsFileWriter(results)

class BestFileWriterFactory(DefaultObjectFactory):

  def __init__(self):
      super().__init__(DefaultResultsFileWriterCreator())


MatcalFileWriterFactory = BestFileWriterFactory()
