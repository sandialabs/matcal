import argparse
from matcal.core.data import convert_dictionary_to_data
from matcal.core.data_importer import FileData
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, MultiSelect, FileInput, Toggle, CustomJS, Div
from bokeh.plotting import figure
from base64 import b64decode
import pandas as pd
import io

from matcal.core.serializer_wrapper import matcal_load   
from matcal.core.surrogates import MatCalMonolithicPCASurrogate, load_matcal_surrogate

COLUMN_WIDTH=600

def parse_parameter_ranges(surrogate_package):
    param_info = surrogate_package[0]
    return param_info[0]


def get_source_data(args):
    if args.training_data != None:
        source_data = matcal_load(args.training_data)[1]
    else:
        source_data = None
    return source_data


def get_arguments():
    arg_message = _get_argparser_message()
    parser = argparse.ArgumentParser(description=arg_message,
                                     conflict_handler='resolve')
    parser.add_argument('-s', "--surrogate_file", type=str, 
                help="File path to MatCal surrogate export file.", default=None, required=True)
    parser.add_argument('-t', "--training_data", type=str,
                help="File path to training data", default=None)

    return parser.parse_args()

def _get_argparser_message():
    arg_message = "Interactive surrogate visualization utility for MatCal. "
    arg_message += "Takes in surrogates trained in MatCal, "
    arg_message += "to allow for easy analysis of surrogate responses."
    return arg_message

def _process_parameter_information(parameter_ranges, p_order):
    bottom = []
    top = []
    initial = []
    parameter_order = []
    for name in p_order:
        values = parameter_ranges[name]
        parameter_order.append(name)
        bottom.append(values[0])
        top.append(values[1])
        average = (values[0] + values[1]) / 2
        initial.append(average)
    return p_order, initial, bottom, top

def _make_initial_surrogate_inputs(param_ranges, surrogate):
    p_init = []
    for field in surrogate.parameter_order:
        low, high = param_ranges[field]
        p_init.append((low+high)/2)
    return p_init

def _make_data_for_plots(initial_param_values, surrogate):
    prediction = surrogate(np.array(initial_param_values).reshape(1,-1))
    y_fields = list(prediction.keys())
    x_field = surrogate.independent_field
    x = prediction[x_field]
    y = prediction[y_fields[0]]
    sources = []
    sources.append(ColumnDataSource(data=dict(x=[x], y=[y], lc=['black'], legend_names=[y_fields[0]])))
    sources.append(ColumnDataSource(data=dict(x=[x], y=[y], lc=['black'], legend_names=[y_fields[0]])))
    return sources, prediction

def _make_plots(indep_field):
    plots = []
    plots.append(figure(title="Surrogate Plot: A", x_axis_label=indep_field, y_axis_label='', 
        height=COLUMN_WIDTH, width=COLUMN_WIDTH, tools="crosshair,pan,reset,save,wheel_zoom"))
    plots.append(figure(title="Surrogate Plot: B", x_axis_label=indep_field, y_axis_label='',
        height=COLUMN_WIDTH, width=COLUMN_WIDTH, tools="crosshair,pan,reset,save,wheel_zoom"))
    return plots


def run_interactive_surrogate(parameter_ranges, surrogate, source_data):
    sliders = _make_sliders(parameter_ranges, surrogate)
    initial_param_values = _make_initial_surrogate_inputs(parameter_ranges, surrogate)
    plot_sources, prediction = _make_data_for_plots(initial_param_values, surrogate)
    plots = _make_plots(surrogate.independent_field)
    plotting_selectors = _make_plotting_selectors(surrogate, prediction)
    file_sources = [None, None]
    file_data = [None, None]

    for plot_idx, plot in enumerate(plots):
        plot.multi_line(xs='x', ys='y', line_width=3, line_alpha=0.9, source=plot_sources[plot_idx],
            line_color='lc', legend_field='legend_names')
        
    def _update_prediction():      
        params = []
        for s in sliders:
            params.append(s.value)        
        prediction = surrogate(np.array(params).reshape(1,-1))
        return prediction

    def _generate_data_update(plot_idx, data_source):
        current_fields = plotting_selectors[plot_idx].value
        x = []
        y = []
        color_wheel = _get_color_wheel()
        selected_colors = []
        legend = []
        for field in current_fields:
            x.append(data_source[surrogate.independent_field])
            try:
                new_y_array = data_source[field].flatten()
            except KeyError:
                _alert_for_missing_field(message_display[plot_idx], field)
                new_y_array = np.zeros_like(data_source[surrogate.independent_field])
            y.append(new_y_array)
            selected_colors.append(color_wheel.pop())
            legend.append(field)
        return x,y,selected_colors,legend

    def _alert_for_missing_field(current_div, missing_field_name:str):
        message = f"<b>ERROR</b>:Field not found: {missing_field_name}"
        current_div.text = message


    def _generate_source_data_update(plot_idx, data_source):
        current_fields = plotting_selectors[plot_idx].value
        x = []
        y = []
        color_wheel = _get_color_wheel()
        selected_colors = []
        legend = []
        for field in current_fields:
            if field == surrogate.independent_field:
                data_array = np.array([surrogate.prediction_locations])
            else:
                data_array = np.array(data_source[field])
            n_samples = data_array.shape[0]
            for samp_idx in range(n_samples):
                x.append(surrogate.prediction_locations)
                y.append(data_array[samp_idx, :].flatten())
        return x,y,selected_colors,legend


    def _update_surro_plot(plot_idx):
        prediction = _update_prediction()
        x, y, selected_colors, legend = _generate_data_update(plot_idx, prediction)

        plot_sources[plot_idx].data = dict(x=x, y=y, lc=selected_colors, legend_names=legend) 
        if file_data[plot_idx] != None:
            x, y, selected_colors, legend = _generate_data_update(plot_idx, file_data[plot_idx])
            update_dict = dict(x=x, y=y, lc=selected_colors, legend_names=legend)
            file_sources[plot_idx].data = update_dict 
        _generate_source_lines(training_data_switches[plot_idx].active, plot_idx)


    def _get_color_wheel():
        return ['wheat', 'green', 'gold','cyan', 'yellow','grey', "orange", "red", 'blue', 'black'] 

    def update_data_A(attrname, old, new):
        plot_idx = 0
        _update_surro_plot(plot_idx) 

    def update_data_B(attrname, old, new):
        plot_idx = 1
        _update_surro_plot(plot_idx) 

    accepted_extensions = ".csv"
    file_input_A = FileInput(name="FI_A", accept=accepted_extensions)
    file_input_B = FileInput(name="FI_B", accept=accepted_extensions)

    training_data_switch_A = Toggle(label="Show Training Data", active=False)
    training_data_switch_B = Toggle(label="Show Training Data", active=False)

    training_data_sources = [None, None]
    training_data_switches = [training_data_switch_A, training_data_switch_B]

    message_display_a = Div()
    message_display_b = Div()
    message_display = [message_display_a, message_display_b]

    def _update_file_plot(plot_idx, data):
        file_data[plot_idx] = data
        x, y, selected_colors, legend = _generate_data_update(plot_idx, data)
        update_dict = dict(x=x, y=y, lc=selected_colors, legend_names=legend)
        if file_sources[plot_idx] != None:
            file_sources[plot_idx].data = update_dict 
        else:
            file_sources[plot_idx] = ColumnDataSource(data = update_dict)
            plots[plot_idx].multi_line(xs='x', ys='y', line_width=3, line_alpha=0.9, 
                source=file_sources[plot_idx], line_color='lc', legend_field='legend_names',
                line_dash='dashed')

    def import_file_data_A(attr, old, new):
        plot_idx = 0
        data = _decode_file_data_to_dict(new, file_inputers[plot_idx])        
        _update_file_plot(plot_idx, data)

    def import_file_data_B(attr, old, new):
        plot_idx = 1
        data = _decode_file_data_to_dict(new, file_inputers[plot_idx])        
        _update_file_plot(plot_idx, data)


    def _decode_file_data_to_dict(new, inputer):
        # Need to use pandas because we don't have access to file path due to 
        # Browser security reasons.
        coded_file_data = new
        readable_file_stream = io.BytesIO(b64decode(coded_file_data))
        data_dict = {}
        df = pd.read_csv(readable_file_stream, skipinitialspace=True, encoding='ascii', delimiter=',')
        for key in df.keys():
            data_dict[key] = df[key].to_numpy()
        return data_dict

    source_data_lines = [None, None]
    def update_source_data_A(active):
        plot_idx = 0
        _generate_source_lines(active, plot_idx)

    def update_source_data_B(active):
        plot_idx = 1
        _generate_source_lines(active, plot_idx)


    def _generate_source_lines(active, plot_idx):
        if active:
            x, y, selected_colors, legend = _generate_source_data_update(plot_idx, source_data)
            scaled_alpha = np.min([.1, 10. / len(x)] )
            update_dict = dict(x=x, y=y)
            if training_data_sources[plot_idx] == None:
                training_data_sources[plot_idx] = ColumnDataSource(update_dict)
                source_data_lines[plot_idx] = plots[plot_idx].multi_line(xs='x', ys='y',
                    line_width=1, line_alpha=scaled_alpha, source=training_data_sources[plot_idx],
                    line_color="black")
            else:
                training_data_sources[plot_idx].data = update_dict
                source_data_lines[plot_idx].visible = True
        else:
            if source_data_lines[plot_idx] != None:
                source_data_lines[plot_idx].visible = False



    file_input_A.on_change('value', import_file_data_A)
    file_input_B.on_change('value', import_file_data_B)
    file_inputers = [file_input_A, file_input_B]

    for s in sliders:
        s.on_change('value', update_data_A, update_data_B)

    training_data_switch_A.on_click(update_source_data_A)
    training_data_switch_B.on_click(update_source_data_B)


    plotting_selectors[0].on_change('value', update_data_A)
    plotting_selectors[1].on_change('value', update_data_B)


    inputs_col = column(*sliders)
    if source_data != None:
        plot_A_col = column(plots[0], training_data_switch_A, message_display_a, 
            file_input_A, plotting_selectors[0])
        plot_B_col = column(plots[1], training_data_switch_B, message_display_b, 
            file_input_B, plotting_selectors[1])
    else:
        plot_A_col = column(plots[0], message_display_a, file_input_A, plotting_selectors[0])
        plot_B_col = column(plots[1], message_display_b, file_input_B, plotting_selectors[1])


    curdoc().add_root(row(inputs_col, plot_A_col, plot_B_col, width=COLUMN_WIDTH))
    curdoc().title = "Interactive Surrogate"

def _make_plotting_selectors(surrogate, prediction):
    y_ms_menu = []
    for key in prediction.keys():
        y_ms_menu.append((key, key))
    multiselect_A = MultiSelect(value=[surrogate.independent_field], options=y_ms_menu)
    multiselect_B = MultiSelect(value=[surrogate.independent_field], options=y_ms_menu)
    multi_selectors = [multiselect_A, multiselect_B]
    return multi_selectors


def _make_sliders(parameter_ranges, surrogate):
    params_order, params_init, params_bottom, params_top = _process_parameter_information(parameter_ranges, surrogate.parameter_order)
    sliders = []
    n_steps = 100
    for name, val_0, val_min, val_max in zip(params_order, params_init, params_bottom, params_top):
        step_size = (val_max - val_min) / n_steps
        sliders.append(Slider(title=name, value=val_0, start=val_min, end=val_max, step=step_size))
    return sliders

args = get_arguments()
surrogate_file = args.surrogate_file
surrogate_package = matcal_load(surrogate_file)[1]
surrogate = load_matcal_surrogate(surrogate_file)
parameter_ranges = parse_parameter_ranges(surrogate_package)
source_data = get_source_data(args)
run_interactive_surrogate(parameter_ranges, surrogate, source_data)
