#!/bin/bash

base_files=(
"config_models_online_overlap_256.yaml"
"config_models_online_overlap_128.yaml"
"config_models_online_cifar5m_overlap_256.yaml"
"config_models_online_cifar5m_overlap_128.yaml"
)

for base_yaml_file in "${base_files[@]}";
do
  report_log="report_offline_f"
  is_offline=True
  python3 main.py --cf "${base_yaml_file}" --audit.report_log "${report_log}" --audit.offline ${is_offline}

  report_log="report_online_f"
  is_offline=False
  python3 main.py --cf "${base_yaml_file}" --audit.report_log "${report_log}" --audit.offline ${is_offline}
done

#report_log="report_population"
#python3 main.py --cf "${base_yaml_file}" --audit.report_log "${report_log}" --audit.offline ${is_offline}