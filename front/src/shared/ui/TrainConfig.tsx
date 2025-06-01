import { Button, Flex, HStack, Text, VStack } from '@chakra-ui/react'
import { useMemo } from 'react'

import { ITrain } from "../types";
import { getDate, getStatus } from '@/pages/Training';
import { LineChart } from '@/shared/ui'
import { getFile } from '../api/training';

const TrainConfig = (configuration: ITrain) => {
  const chartData = useMemo(() => {
    return configuration?.result_metrics
  }, [configuration])

  const DownloadAction = (type: string) => {
    getFile(configuration.id!, type).then((response) => {
      console.log('resp:', response);
      
      const a = document.createElement("a");
      document.body.appendChild(a);

      const json = JSON.stringify(response.data),
        blob = new Blob([json], { type: "octet/stream" }),
        url = window.URL.createObjectURL(blob)
      
      console.log(url);
      
      a.href = url
      switch (type) {
        case 'pt':
          a.download = 'best.pt'
          break;
        default:
          a.download = 'best.onnx'
      }

      a.click()
      window.URL.revokeObjectURL(url)
    }
    )
  }
  return (
    <Flex w="100%"  pt='10px' mt='30px' borderTopWidth="1px" borderColor=''>
      <VStack alignItems="flex-start">
        <Text>Выбранная модель: {configuration?.model}</Text> 
        <Text>
          Статус обучения: {getStatus(configuration?.status) || 'нет'}
        </Text>
        {/* <Text>
          Время создания: {getDate(configuration?.created_at) || 'нет'}
        </Text> */}
        <Text as="b" pt="20px">
          Конфигурация
        </Text>
        <Text>
          Размер батча: {configuration?.training_conf?.batch || 'нет'}
        </Text>
        {/* <Text>
          Размер изображения: {configuration?.training_conf?.imgsz || 'нет'}
        </Text> */}
        <Text>
          Метод оптимизации:{' '}
          {configuration?.training_conf?.optimizer || 'нет'}
        </Text>
      </VStack>
      <VStack w="70%">
        {chartData ? <LineChart {...chartData} /> : <></>}
        <Button onClick={() => DownloadAction("pt")}>Скачать веса в формате .pt</Button>
        <Button onClick={() => DownloadAction("onnx")}>Скачать веса в формате .onnx</Button>
      </VStack>
    </Flex>
  )
}

export default TrainConfig;