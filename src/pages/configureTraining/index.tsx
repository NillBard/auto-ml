import { Button, Flex, Input, Select, Text, VStack } from '@chakra-ui/react'
import { useState } from 'react'
import { ITrainCreate } from '../../types/train.ts'
import {
  createTrainingConfiguration,
  startLearning,
  uploadDataset,
} from '../../api/api.ts'
import { useNavigate } from 'react-router-dom'

const ConfigureTraining = () => {
  const [name, setName] = useState('')
  const [model, setModel] = useState('')
  const [epoch, setEpoch] = useState(0)
  const [batch, setBatch] = useState(16)
  const [imgsz, setImgsz] = useState(640)
  const [optimizer, setOptimizer] = useState('auto')
  const [classes, setClasses] = useState<string[]>([])

  const [file, setFile] = useState<File | null>(null)
  const navigate = useNavigate()

  const ButtonAction = () => {
    const conf: ITrainCreate = {
      name: name,
      model: model,
      epochs: epoch,
      batch: batch,
      imgsz: imgsz,
      optimizer: optimizer,
      class_names: classes,
    }

    createTrainingConfiguration(conf)
      .then(({ data }) => {
        if (file && data) {
          uploadDataset(data.id, file).then(() => {
            startLearning(data.id).then(() => {
              console.log('Vsee ok')
            })
          })
        }
      })
      .catch((e) => console.log(e))
      .finally(() => {
        navigate('/training')
      })
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
          <option value="yolov8s">yolov8s</option>
          <option value="yolov8m">yolov8m</option>
          <option value="yolov8l">yolov8l</option>
          <option value="yolov8x">yolov8x</option>
        </Select>
        <Text>Выберите платформу обучения</Text>
        <Select placeholder="Выберите платформу">
          <option value="cpu">Процессор</option>
          <option value="0">Видеокарта</option>
        </Select>
        <Text pt="25px">Задайте конфигурацию обучения:</Text>
        <Text>Укаажите имена классов в порядке разметки</Text>
        <Input
          placeholder="Название классов"
          onChange={(e) => {
            setClasses(e.target.value.split(','))
          }}
        ></Input>
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
          <option value="SGD">SGD</option>
          <option value="Adam">Adam</option>
          <option value="Adamax">Adamax</option>
          <option value="AdamW">AdamW</option>
          <option value="NAdam">NAdam</option>
          <option value="RAdam">RAdam</option>
          <option value="RMSProp">RMSProp</option>
        </Select>
        <Text>Загрузить файл</Text>
        <Input
          type="file"
          onChange={(e) => {
            if (e.target.files) setFile(e.target.files[0])
          }}
        ></Input>
        <Button onClick={ButtonAction}>Запустить обучение</Button>
      </VStack>
    </Flex>
  )
}

export default ConfigureTraining
