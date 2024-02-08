import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process log file and create Excel file.')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to the log file.')
    parser.add_argument("--job_num", type=str, default=None)
    return parser.parse_args()

# 定义自定义排序规则的函数
def custom_sort_key(row):
    # 根据 'model id' 和 'dtype' 的自定义规则返回一个元组
    model_id_sort_order = {
        'EleutherAI-gpt-j-6b':1,
        'meta-llama-Llama-2-7b-hf':2,
        'meta-llama-Llama-2-13b-hf':3,
        'meta-llama-Llama-2-70b-hf':4,
        'EleutherAI-gpt-neox-20b':5,
        'tiiuae-falcon-40b':6,
        'bigscience-bloom-1b7': 7,
        'facebook-opt-1.3b': 8,
        'facebook-opt-30b': 9,
        'Salesforce-codegen-2B-multi': 10,
        'THUDM-chatglm3-6b': 11,
        'THUDM-chatglm2-6b': 12,
        'baichuan-inc-Baichuan2-13B-Chat':13,
        'Baichuan-inc-Baichuan2-13B-Chat':14,
        'baichuan-inc-Baichuan2-7B-Chat':15,
        'Baichuan-inc-Baichuan2-7B-Chat':16,
        'baichuan-inc-Baichuan-13B-Chat':17,
        'Baichuan-inc-Baichuan-13B-Chat':18,
        'bigcode-starcoder': 19,
        'google-flan-t5-xl': 20,
        'mistralai-Mistral-7B-v0.1':21,
        'mosaicml-mpt-7b':22,
        'stabilityai-stablelm-2-1-6b': 23,
    }

    dtype_sort_order = {
        'bfloat16': 1,
        'static-fp32-int8': 2,
        'static-bf16-int8': 3,
        'woq-int8':4,
        'woq-int4': 5,
        'woq-int4-bf16': 6,
        'ds-bfloat16':7,
        'ds-woq-int8': 8,
        'ds-woq-int4': 9,

    }

    input_sort_order = {
        '128-32-1':1,
        '512-32-1':2,
        '1024-32-1':3,
        '2016-32-1':4,
    }

    return (model_id_sort_order.get(row['model id'], float('inf')),
            dtype_sort_order.get(row['dtype'], float('inf')),
            input_sort_order.get(row['intput-output-bs'], float('inf')))

def main():
    # 解析命令行参数
    args = parse_arguments()
    # 创建空的DataFrame
    df = pd.DataFrame(columns=['model id', 'dtype', 'intput-output-bs', 'beam', 'rank', 'total latency','log'])

    # 读取 log 文件并逐行写入 Excel
    with open(args.log_dir, 'r') as log_file:
        for line in log_file:
            # 去掉 "llm_default_"，然后以第一个 "," 为分界将每一行分成两部分
            parts = line.replace('llm_default_', '').strip().split(',', 1)

            # 前一部分使用 "_" 分割
            first_half = parts[0].split('_')

            # 后一部分使用 "," 分割
            second_half = parts[1].split(',')

            # 如果 first_half[4] 的值为 "False"，替换为4，若为“True“，否则为1
            beam_value = 1 if first_half[4] != 'False' else 4

            # 创建字典，将数据写入 DataFrame
            row_data = {
                'model id': first_half[0],
                'dtype': first_half[1],
                'intput-output-bs': first_half[2],
                'beam': beam_value,
                'rank': first_half[6],
                'total latency': second_half[0],
                'memory': second_half[5]
            }

            df = df._append(row_data, ignore_index=True)

    # 根据自定义排序规则排序 DataFrame
    df['sort_key'] = df.apply(custom_sort_key, axis=1)
    df.sort_values(by='sort_key', inplace=True)
    df.drop(columns='sort_key', inplace=True)

    # 将DataFrame写入Excel文件
    excel_file_path = args.job_num + '_output.xlsx'
    df.to_excel(excel_file_path, index=False)

    print(f"Excel文件已创建：{excel_file_path}")

if __name__ == "__main__":
    main()




