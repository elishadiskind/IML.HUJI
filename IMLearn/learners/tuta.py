import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


def load_data(file_path: str) -> pd.DataFrame:
    data_dict = pd.read_excel(file_path, sheet_name=[1, 2, 3, 4, 5], date_parser=['תאריך'])
    data_dict['Leumi'] = data_dict.pop(1)
    data_dict['Hapoalim'] = data_dict.pop(2)
    data_dict['Discount'] = data_dict.pop(3)
    data_dict['Benleumi'] = data_dict.pop(4)
    data_dict['Mizrahi'] = data_dict.pop(5)
    return data_dict


if __name__ == '__main__':
    data_dict = load_data(r"C:\Users\elish\OneDrive\Documents\GitHub\IML\IML.HUJI\datasets\data_history.xlsx")
    madad_df = pd.DataFrame
    for bank in data_dict:
        data_dict[bank].append()
    #print(data_dict['Leumi'].iloc[:,0:1])
