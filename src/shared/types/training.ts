export interface ITrainConfiguration {
    time: number
    batch: number
    imgsz: number
    epochs: number
    patience: number
    optimizer: string
}

export type ResultMetricsKeys =
    | 'recall'
    | 'mAP50'
    | 'mAP50-95'
    | 'precision'
    | 'val/box_loss'
    | 'val/cls_loss'
    | 'val/dfl_loss'
    | 'train/box_loss'
    | 'train/cls_loss'
    | 'train/dfl_loss'

export type Metric = Record<string, number>

export type Metrics = Record<ResultMetricsKeys, Metric>

export interface ITrain {
    id: number
    name: string
    model: string
    status: string
    dataset_d3_url: string
    weight_s3_url: string | null
    created_at: string
    training_conf: ITrainConfiguration
    result_metrics: Metrics
}

export interface IDataset {
    id: number
    name: string
    created_date: string
    status: string
}

export interface ITrainCreate {
    name: string
    model: string
    epochs: number
    batch: number
    imgsz: number
    optimizer: string
    task_type: string
    class_names: string[]
    device: string
    dataset_id: number
}

export interface ICameraInfo {
    location: string
    login: string
    password: string
}

export interface IMlModel {
  label: string,
  value: string,
}
