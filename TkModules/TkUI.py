
import dearpygui.dearpygui as dpg
import time
import torch

class TkUI():

    FIT_AXIS_TIMEOUT = 60.0
    _last_fit_axis_time = dict()

    @staticmethod
    def need_fit_axis(series_tag:str):
        return True
        if series_tag in TkUI._last_fit_axis_time:
            elapsed_time = time.time() - TkUI._last_fit_axis_time[series_tag] 
            if elapsed_time > TkUI.FIT_AXIS_TIMEOUT:
                TkUI._last_fit_axis_time[series_tag] = time.time()
                return True
            else:
                return False
        else:
            TkUI._last_fit_axis_time[series_tag] = time.time()
            return True

    @staticmethod
    def set_series( x_axis_tag:str, y_axis_tag:str, series_tag:str, series:list):
        dpg.set_value( series_tag, [[i for i in range(0, len(series))], series]) 
        if TkUI.need_fit_axis( series_tag ):
            dpg.fit_axis_data( x_axis_tag )
            dpg.fit_axis_data( y_axis_tag )

    @staticmethod
    def set_series_with_labels( x_axis_tag:str, y_axis_tag:str, series_tag:str, series:list, labels:list):
        dpg.set_value( series_tag, [labels, series])
        #dpg.set_value( series_tag, [[i for i in range(0, len(series))], series]) 
        if TkUI.need_fit_axis( series_tag ):
            dpg.fit_axis_data( x_axis_tag )
            dpg.fit_axis_data( y_axis_tag )

    @staticmethod
    def set_series_from_tensor( x_axis_tag:str, y_axis_tag:str, series_tag:str, t:torch.Tensor, batch_id:int):        
        series = []
        if len(t.size()) > 2:
            for i in range(t.size(dim=2)):
                series.append( t[batch_id,0,i].item() )
        else:
            for i in range(t.size(dim=1)):
                series.append( t[batch_id,i].item() )
        dpg.set_value( series_tag, [[i for i in range(0, len(series))], series])
        if TkUI.need_fit_axis( series_tag ):
            dpg.fit_axis_data( x_axis_tag )
            dpg.fit_axis_data( y_axis_tag )