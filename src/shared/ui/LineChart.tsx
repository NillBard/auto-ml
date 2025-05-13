import { Metrics } from '@/shared/types'
import { Line } from 'react-chartjs-2'
import { HStack, VStack } from '@chakra-ui/react'
import { ChartData, Point } from 'chart.js'

export const LineChart = (metrics: Metrics) => {
  const data:
    | ChartData<'line', (number | Point | null)[], unknown>[]
    | {
      labels: string[]
      datasets: {
        label: string
        data: number[]
        borderColor: string
        backgroundColor: string
      }[]
    }[] = []
  const getLine = () => {
    Object.entries(metrics).map(([key, value]) => {
      console.log(key, value)
      const chartData = {
        labels: Object.keys(value || {}),
        datasets: [
          {
            label: key,
            data: Object.values(value || {}),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
        ],
      }
      
      if(!['best_metrics', 'epochs'].includes(key)) {
        data.push(chartData)
      }
    })
  }

  getLine()

  console.log('data: ', data);
  
  return (
    <VStack w="100%">
      {data.map((_, index) => {
        if(index <= data.length/2 - 1 ) {
          console.log('index', index);
          
          return (
            (
              <HStack w="45%" justifyContent="center">
                <Line data={data[index*2]} />
                <Line data={data[index]} />
              </HStack>
            )
          )
        }
      })}
      {/* <HStack w="30%" justifyContent="center">
        <Line data={data[0]} />
        <Line data={data[1]} />
      </HStack>
      <HStack w="30%" justifyContent="center">
        <Line data={data[2]} />
        <Line data={data[3]} />
      </HStack>
      <HStack w="30%" justifyContent="center">
        <Line data={data[4]} />
        <Line data={data[5]} />
      </HStack>
      <HStack w="30%" justifyContent="center">
        <Line data={data[6]} />
        <Line data={data[7]} />
      </HStack>
      <HStack w="30%" justifyContent="center">
        <Line data={data[8]} />
        <Line data={data[9]} />
      </HStack> */}
    </VStack>
  )
}
