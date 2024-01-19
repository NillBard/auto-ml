import axios from './axios.ts'

import { ITrain, ITrainCreate } from '../types/train.ts'

export async function testConnection(source: string) {
  return await axios.post('/pipeline/test', {
    source: source,
  })
}

export async function getTrainingConfigurations() {
  return await axios.get<[ITrain]>('/train/')
}

export async function createTrainingConfiguration(configuration: ITrainCreate) {
  return await axios.post('/train/', configuration)
}
