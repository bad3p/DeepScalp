
import dearpygui.dearpygui as dpg
import torch

class TkUI():

    @staticmethod
    def set_series( x_axis_tag:str, y_axis_tag:str, series_tag:str, series:list):
        dpg.set_value( series_tag, [[i for i in range(0, len(series))], series]) 
        dpg.fit_axis_data( x_axis_tag )
        dpg.fit_axis_data( y_axis_tag )

    @staticmethod
    def set_series_with_labels( x_axis_tag:str, y_axis_tag:str, series_tag:str, series:list, labels:list):
        dpg.set_value( series_tag, [labels, series])
        #dpg.set_value( series_tag, [[i for i in range(0, len(series))], series]) 
        dpg.fit_axis_data( x_axis_tag )
        dpg.fit_axis_data( y_axis_tag )

    @staticmethod
    def set_series_from_tensor( x_axis_tag:str, y_axis_tag:str, series_tag:str, t:torch.Tensor, batch_id:int):
        series = []
        if len(t.size()) > 2:
            for i in range(t.size(dim=2)):
                series.append( t[batch_id,0,i] )
        else:
            for i in range(t.size(dim=1)):
                series.append( t[batch_id,i] )
        dpg.set_value( series_tag, [[i for i in range(0, len(series))], series])
        dpg.fit_axis_data( x_axis_tag )
        dpg.fit_axis_data( y_axis_tag )