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

import { ITrain, ITrainProject } from '@/shared/types'
import { getFile, getTrainingResults } from '@/shared/api'
import { getDate, getStatus } from './Training.tsx'
import { LineChart } from '@/shared/ui'
import TrainConfig from '@/shared/ui/TrainConfig.tsx'

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
  const [configuration, setConfiguration] = useState<ITrainProject>({} as ITrainProject)

  const { configurationId } = useParams()

  useEffect(() => {
    if (configurationId) {
      getTrainingResults(configurationId).then(({ data }) =>
        {
          setConfiguration(data)
        }
      )
    } else {
      setConfiguration({} as ITrainProject)
    }
  }, [configurationId])

  const bestModel = () => {
    console.log(configuration.trains);
    
    const model = configuration?.trains?.find(el => el.id === configuration.best_model_id);
    console.log('model', model);
    
    return model
  }

  const trains = useMemo(() => {
    const model = bestModel()
    let trains = [];
    if (model !== undefined) {
      trains = [model, ...configuration.trains.filter(el => el.id !== configuration.best_model_id)]
    } else {
      trains = configuration.trains
    }

    console.log('trains', trains);
    
    return trains
  }, [configuration.trains])
  // const chartData = useMemo(() => {
  //   return configuration?.result_metrics
  // }, [configuration])


  // const DownloadAction = (type: string) => {
  //   getFile(configurationId!, type).then((response) => {
  //     const a = document.createElement("a");
  //     document.body.appendChild(a);
  //     const json = JSON.stringify(response.data),
  //       blob = new Blob([json], { type: "octet/stream" }),
  //       url = window.URL.createObjectURL(blob)
  //     a.href = url
  //     switch (type) {
  //       case 'pt':
  //         a.download = 'dataset.pt'
  //         break;
  //       default:
  //         a.download = 'dataset.onnx'
  //     }

  //     a.click()
  //     window.URL.revokeObjectURL(url)
  //   }
  //   )
  // }

  // console.log(configuration);
  
  
  return (
    <Flex flexDirection="column" w="100%" pl='10'>
      <Flex w="100%" pt="30px" justifyContent="center" flexDirection="column">
        <VStack alignItems="flex-start">
          <HStack>
            <button onClick={() => navigator('/training')} > BACK</button>
            <Text as="b">{configuration?.name}</Text>
          </HStack>
          {/* <Text>Выбранная модель: {configuration?.model}</Text> */}
          <Text>
            Статус обучения: {getStatus(configuration?.status) || 'нет'}
          </Text>
          <Text>
            Лучшая модель: {bestModel()?.model || 'нет'}
          </Text>
          <Text>
            Время создания: {getDate(configuration?.created_at) || 'нет'}
          </Text>
          <Text as="b" pt="20px">
            Модели
          </Text>
          {/* <Text>
            Размер батча: {configuration?.training_conf?.batch || 'нет'}
          </Text> */}
          {/* <Text>
            Размер изображения: {configuration?.training_conf?.imgsz || 'нет'}
          </Text> */}
          {/* <Text>
            Метод оптимизации:{' '}
            {configuration?.training_conf?.optimizer || 'нет'}
          </Text> */}
        </VStack>
        {/* <VStack w="50%">
          {chartData ? <LineChart {...chartData} /> : <></>}
          <Button onClick={() => DownloadAction("pt")}>Скачать веса в формате .pt</Button>
          <Button onClick={() => DownloadAction("onnx")}>Скачать веса в формате .onnx</Button>
        </VStack> */}
        {
          trains && trains.map(train => (
            <TrainConfig {...train}/>
          ))
        }
      </Flex>
    </Flex>
  )
}

export default ConfigurePage
