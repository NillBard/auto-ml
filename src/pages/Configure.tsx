import { Button, Flex, HStack, Text, VStack } from '@chakra-ui/react'
import { useNavigate, useParams } from 'react-router-dom'
import { useEffect, useMemo, useState } from 'react'

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'

import { ITrain } from '../types/train.ts'
import { getFile, getTrainingResults } from '../api/api.ts'
import { getDate, getStatus } from './Training.tsx'
import LineChart from '../components/lineChart/index.tsx'
import { ArrowBackIcon } from '@chakra-ui/icons'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)


const ConfigurePage = () => {
  const navigator = useNavigate()
  const [configuration, setConfiguration] = useState<ITrain>({} as ITrain)

  const { configurationId } = useParams()

  useEffect(() => {
    if (configurationId) {
      getTrainingResults(configurationId).then(({ data }) =>
        setConfiguration(data)
      )
    } else {
      setConfiguration({} as ITrain)
    }
  }, [configurationId])

  const chartData = useMemo(() => {
    return configuration?.result_metrics
  }, [configuration])


  const DownloadAction = (type: string) => {
    getFile(configurationId!, type).then((response) => {
      const a = document.createElement("a");
      document.body.appendChild(a);
      const json = JSON.stringify(response.data),
        blob = new Blob([json], { type: "octet/stream" }),
        url = window.URL.createObjectURL(blob)
      a.href = url
      switch (type) {
        case 'pt':
          a.download = 'dataset.pt'
          break;
        default:
          a.download = 'dataset.onnx'
      }

      a.click()
      window.URL.revokeObjectURL(url)
    }
    )
  }

  return (
    <Flex flexDirection="column" w="100%">
      <Flex w="100%" pt="30px" justifyContent="center">
        <VStack alignItems="flex-start">
          <HStack>
            <ArrowBackIcon onClick={() => navigator('/training')} />
            <Text as="b">{configuration?.name}</Text>
          </HStack>
          <Text>Выбранная модель: {configuration?.model}</Text>
          <Text>
            Статус обучения: {getStatus(configuration?.status) || 'нет'}
          </Text>
          <Text>
            Время создания: {getDate(configuration?.created_at) || 'нет'}
          </Text>
          <Text as="b" pt="20px">
            Конфигурация
          </Text>
          <Text>
            Размер батча: {configuration?.training_conf?.batch || 'нет'}
          </Text>
          <Text>
            Размер изображения: {configuration?.training_conf?.imgsz || 'нет'}
          </Text>
          <Text>
            Метод оптимизации:{' '}
            {configuration?.training_conf?.optimizer || 'нет'}
          </Text>
        </VStack>
        <VStack w="50%">
          {chartData ? <LineChart {...chartData} /> : <></>}
          <Button onClick={() => DownloadAction("pt")}>Скачать веса в формате .pt</Button>
          <Button onClick={() => DownloadAction("onnx")}>Скачать веса в формате .onnx</Button>
        </VStack>
      </Flex>
    </Flex>
  )
}

export default ConfigurePage
