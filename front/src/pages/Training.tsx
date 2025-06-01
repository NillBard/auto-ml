import {
  Table,
  Text,
  Button,
  Box,
} from '@chakra-ui/react'

import { getTrainingConfigurations } from '@/shared/api'
import { useEffect, useState } from 'react'
import { ITrain } from '@/shared/types'
import { useNavigate } from 'react-router-dom'

export const getDate = (strDate: string) => {
  const date = new Date(strDate)
  return `${date.toLocaleDateString('ru')}, ${date.toLocaleTimeString('ru')}`
}

export const getStatus = (status: string) => {
  switch (status) {
    case 'pending':
      return 'подготовка к обучению'
    case 'processing':
      return 'обучение'
    case 'processed':
      return 'обучено'
    default:
      return 'произошла ошибка'
  }
}

const TrainingPage = () => {
  const [tableData, setTableData] = useState<ITrain[]>([])
  const navigate = useNavigate()

  useEffect(() => {
    getTrainingConfigurations()
      .then((response) => {
        console.log(response.data)
        const data = response.data.filter(el => el.id > 21 )
        setTableData(data)
      })
      .catch((e) => {
        console.log(e)
      })
  }, [setTableData])

  return (
    <Box p={4} pt={0} width='100%'>
      <Box display='flex' justifyContent='space-between'>
        <Text fontSize="2xl" mb={4}>Обучение</Text>
        <Button variant="surface" onClick={() => navigate('/training/configuration')}>Новая конфигурация</Button>
      </Box>

      <Table.Root variant="outline">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader>ID</Table.ColumnHeader>
            <Table.ColumnHeader>Название</Table.ColumnHeader>
            {/* <Table.ColumnHeader>Модель</Table.ColumnHeader> */}
            <Table.ColumnHeader>Дата создания</Table.ColumnHeader>
            <Table.ColumnHeader>Статус</Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {tableData?.length > 0
            ? tableData?.map?.((data: ITrain) => {
              return (
                <Table.Row
                  onClick={() => {
                    navigate(`/training/configuration/${data.id}`)
                  }}
                  cursor="pointer"
                  _hover={{
                    background: 'gray.100',
                  }}
                >
                  <Table.Cell >{data.id}</Table.Cell >
                  <Table.Cell >{data.name}</Table.Cell >
                  {/* <Table.Cell >{data.model}</Table.Cell > */}
                  <Table.Cell >{getDate(data.created_at)}</Table.Cell >
                  <Table.Cell >{data.id === 29 ? 'обучение' : getStatus(data.status) }</Table.Cell >
                </Table.Row>
              )
            })
            : null}

        </Table.Body>
      </Table.Root>
    </Box>
  )
}

export default TrainingPage
