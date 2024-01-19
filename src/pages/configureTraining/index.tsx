import { Button, Flex, Input, Select, Text, VStack } from '@chakra-ui/react'
import { useState } from 'react'
import { ITrainCreate } from '../../types/train.ts'
import { createTrainingConfiguration } from '../../api/api.ts'

const ConfigureTraining = () => {
  const [name, setName] = useState('')
  const [model, setModel] = useState('')
  const [epoch, setEpoch] = useState(0)
  const [batch, setBatch] = useState(16)
  const [imgsz, setImgsz] = useState(640)
  const [optimizer, setOptimizer] = useState('auto')

  const ButtonAction = () => {
    const configuration: ITrainCreate = {
      name: name,
      model: model,
      epochs: epoch,
      batch: batch,
      imgsz: imgsz,
      optimizer: optimizer,
    }

    createTrainingConfiguration(configuration)
      .then((response) => console.log(response))
      .catch((e) => console.log(e))
  }

  return (
    <Flex pt="30px" pl="30px">
      <VStack>
        <Text>Название проекта: </Text>
        <Input
          placeholder="Название проекта"
          onChange={(e) => {
            setName(e.target.value)
          }}
        ></Input>
        <Text>Выберите необходимую для обучения модель:</Text>
        <Select
          placeholder="Выберите модель"
          onChange={(e) => {
            setModel(e.target.value)
          }}
        >
          <option value="yolov8n">yolov8n</option>
        </Select>
        <Text>Задайте конфигурацию обучения:</Text>
        <Text>Количество эпох обучения</Text>
        <Input
          placeholder="Эпохи"
          onChange={(e) => {
            setEpoch(Number(e.target.value))
          }}
        ></Input>
        <Text>Размер батча</Text>
        <Input
          placeholder="Батч"
          onChange={(e) => {
            setBatch(Number(e.target.value))
          }}
        ></Input>
        <Text>Размер изображения</Text>
        <Input
          placeholder="Размер"
          onChange={(e) => {
            setImgsz(Number(e.target.value))
          }}
        ></Input>
        <Text>Оптимизатор</Text>
        <Select
          placeholder="Выбрать оптимизатор"
          onChange={(e) => {
            setOptimizer(e.target.value)
          }}
        >
          <option value="auto">auto</option>
        </Select>
        <Button onClick={ButtonAction}>Запустить обучение</Button>
      </VStack>
      <VStack pl="40px">
        <Text>Загрузить файл</Text>
        <Input type="file"></Input>
      </VStack>
    </Flex>
  )
}

export default ConfigureTraining
