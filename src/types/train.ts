export interface ITrainConfiguration {
  time: number
  batch: number
  imgsz: number
  epochs: number
  patience: number
  optimizer: string
}

export interface ITrain {
  id: number
  name: string
  model: string
  status: string
  dataset_d3_url: string
  weight_s3_url: string | null
  created_at: string
  training_conf: ITrainConfiguration
}

export interface ITrainCreate {
  name: string
  model: string
  epochs: number
  batch: number
  imgsz: number
  optimizer: string
  class_names: string[]
}
