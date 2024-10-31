# Отбор на проекты Controllable GenAI (AIRI) и BayesGroup (ВШЭ) 2024-2025

## Setup окружения
```
python3 -m venv .genai_task
source .genai_task/bin/activate
pip install -r requirements.txt
```

## Работа с конфигурациями
Шаблоны конфигов находятся в папке `configs`. Для работы с ними необходимо указать путь к `train` и `test` директориям
с данными в полях `input_train_dir` и `input_test_dir` соответственно.
Если присутствует чекпоинт, то стоит также изменить поле `checkpoint_path`
Обучение тестировалось на предоставленном [датасете](https://drive.google.com/file/d/1mDVacZimuy-4hWNylqlBDAO740Hup3W5/view)

## Обучение
```
python3 inference.py exp.config_path=<path_to_train_config> \
                     exp.use_wandb=False \
                     data.input_train_dir=<path_to_train> \
                     data.input_val_dir=<path_to_val> \ 
                     train.checkpoint_path=<path_to_checkpoint> \ 
                     model_args.learn_sigma=<True in case of VLB + MSE, False otherwise>
```

## Применение
```
python3 inference.py exp.config_path=<path_to_inference_config> \
                     exp.use_wandb=False \
                     data.input_train_dir=<path_to_train> \
                     data.input_val_dir=<path_to_val> \ 
                     train.checkpoint_path=<path_to_checkpoint> \ 
                     model_args.learn_sigma=<True in case of VLB + MSE, False otherwise>
```
Инференс модели сгенерирует изображения для каждого класса и поместит их в директорию указанную в `exp.exp_dir`

## Чекпоинты

- Лучшая модель. FID 0.35. Скачать для использования по ссылке [VLB + MSE](https://drive.google.com/file/d/1C8p1sOno6tSargmNO_aBM5ds4as5yZ7z/view?usp=sharing)
   - Для применения необходимо указать параметр `learn_sigma: True`

## Отчет
Отчет находится в этом [файле](./Report.pdf)
