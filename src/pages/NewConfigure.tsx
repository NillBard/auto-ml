import { Button, createListCollection, Flex, Input, Portal, Select, Text, VStack } from '@chakra-ui/react'
import { useEffect, useState } from 'react'
import { IDataset, ITrainCreate } from '@/shared/types'
import {
  createTrainingConfiguration,
  getDatasets,
} from '@/shared/api'
import { useNavigate } from 'react-router-dom'

const NewConfigurePage = () => {
  const [name, setName] = useState('')
  const [model, setModel] = useState('')
  const [epoch, setEpoch] = useState(0)
  const [batch, setBatch] = useState(16)
  const [imgsz, setImgsz] = useState(640)
  const [optimizer, setOptimizer] = useState('auto')
  const [datasets, setDatasets] = useState<IDataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<string>("")

  const navigate = useNavigate()


  useEffect(() => {
    getDatasets()
      .then((response) => {
        console.log(response.data)
        setDatasets(response.data)
      })
      .catch((e) => {
        if (e.response) {
          console.log(e.response.status)
        }
      })
  }, [setDatasets])

  const ButtonAction = () => {
    const conf: ITrainCreate = {
      name: name,
      model: model,
      epochs: epoch,
      batch: batch,
      imgsz: imgsz,
      optimizer: optimizer,
      class_names: [],
      device: "cpu",
      dataset_id: Number(selectedDataset)
    }

    createTrainingConfiguration(conf)
      .then(() => {
        console.log("start_train")
      })
      .catch((e) => {

        console.log(e)
      })
      .finally(() => {
        navigate('/training')
      })
  }

  const models = createListCollection({
    items: [
      { label: "yolov8n", value: "yolov8n" },
      { label: "yolov8s", value: "yolov8s" },
      { label: "yolov8m", value: "yolov8m" },
      { label: "yolov8l", value: "yolov8l" },
      { label: "yolov8x", value: "yolov8x" },
    ],
  })

  const [platform, setPlatforms] = useState('')
  const platforms = createListCollection({
    items: [
      { label: "Процессор", value: "cpu" },
      { label: "Видеокарта", value: "gpu" },
    ],
  })

  const optimizers = createListCollection({
    items: [
      { label: "auto", value: "auto" },
      { label: "SGD", value: "SGD" },
      { label: "Adam", value: "Adam" },
      { label: "Adamax", value: "Adamax" },
      { label: "AdamW", value: "AdamW" },
      { label: "NAdam", value: "NAdam" },
      { label: "RAdam", value: "RAdam" },
      { label: "RMSProp", value: "RMSProp" },
    ],
  })

  const datasetsOption = createListCollection({
    items: datasets ? datasets.map((dataset: IDataset) => (
      { label: dataset.name, value: dataset.id }
    )) : [],
  })

  return (
    <Flex>
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
          <Select.Root collection={models} size="sm" value={[model]}
            onValueChange={(e) => setModel(e.value?.[0])}>
            <Select.HiddenSelect />
            <Select.Control>
              <Select.Trigger>
                <Select.ValueText placeholder="Выберите модель" />
              </Select.Trigger>
              <Select.IndicatorGroup>
                <Select.Indicator />
              </Select.IndicatorGroup>
            </Select.Control>
            <Portal>
              <Select.Positioner>
                <Select.Content>
                  {models.items.map((model) => (
                    <Select.Item item={model} key={model.value}>
                      {model.label}
                      <Select.ItemIndicator />
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Positioner>
            </Portal>
          </Select.Root>
          <Text>Выберите платформу обучения</Text>
          <Select.Root collection={platforms} size="sm" value={[platform]}
            onValueChange={(e) => setPlatforms(e.value?.[0])}>
            <Select.HiddenSelect />
            <Select.Control>
              <Select.Trigger>
                <Select.ValueText placeholder="Выберите платформу" />
              </Select.Trigger>
              <Select.IndicatorGroup>
                <Select.Indicator />
              </Select.IndicatorGroup>
            </Select.Control>
            <Portal>
              <Select.Positioner>
                <Select.Content>
                  {platforms.items.map((item) => (
                    <Select.Item item={item} key={item.value}>
                      {item.label}
                      <Select.ItemIndicator />
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Positioner>
            </Portal>
          </Select.Root>
          <Text pt="25px">Задайте конфигурацию обучения:</Text>
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
          <Select.Root collection={optimizers} size="sm" value={[optimizer]}
            onValueChange={(e) => setOptimizer(e.value?.[0])}>
            <Select.HiddenSelect />
            <Select.Control>
              <Select.Trigger>
                <Select.ValueText placeholder="Выбрать оптимизатор" />
              </Select.Trigger>
              <Select.IndicatorGroup>
                <Select.Indicator />
              </Select.IndicatorGroup>
            </Select.Control>
            <Portal>
              <Select.Positioner>
                <Select.Content>
                  {optimizers.items.map((item) => (
                    <Select.Item item={item} key={item.value}>
                      {item.label}
                      <Select.ItemIndicator />
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Positioner>
            </Portal>
          </Select.Root>
          <Text>Выберите датасет</Text>
          <Select.Root collection={datasetsOption} size="sm" value={[selectedDataset]}
            onValueChange={(e) => setSelectedDataset(e.value?.[0])}>
            <Select.HiddenSelect />
            <Select.Control>
              <Select.Trigger>
                <Select.ValueText placeholder="Датасет" />
              </Select.Trigger>
              <Select.IndicatorGroup>
                <Select.Indicator />
              </Select.IndicatorGroup>
            </Select.Control>
            <Portal>
              <Select.Positioner>
                <Select.Content>
                  {datasetsOption.items.map((item) => (
                    <Select.Item item={item} key={item.value}>
                      {item.label}
                      <Select.ItemIndicator />
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Positioner>
            </Portal>
          </Select.Root>
          <Button onClick={ButtonAction}>Запустить обучение</Button>
        </VStack>
      </Flex>
    </Flex>
  )
}

export default NewConfigurePage
