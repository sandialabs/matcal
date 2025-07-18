from matcal.core.object_factory import ObjectCreator
from matcal.core.best_material_file_writer import DefaultResultsFileWriter, MatcalFileWriterFactory

from matcal.sierra.aprepro_format import make_aprepro_string_from_name_val_pair


class BestApreproMaterialFileWriter(DefaultResultsFileWriter):
    def _get_parameter_lines(self):
        app_lines = []
        for key, value in self.parameters.items():
            line = make_aprepro_string_from_name_val_pair(key, value)
            app_lines.append(line)
        return app_lines
            
class AriaResultsFileWriterCreator(ObjectCreator):
    
  def __call__(self, results):
      return BestApreproMaterialFileWriter(results)

class AdagioResultsFileWriterCreator(ObjectCreator):
    
  def __call__(self, results):
      return BestApreproMaterialFileWriter(results)


MatcalFileWriterFactory.register_creator('aria', AriaResultsFileWriterCreator())
MatcalFileWriterFactory.register_creator('adagio', AdagioResultsFileWriterCreator())