import {
  Table,
  Text,
  Button,
  Link,
  Box,
} from '@chakra-ui/react'

import { getDatasets } from '@/shared/api'
import { useEffect, useState } from 'react'
import { IDataset } from '@/shared/types'
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

const DatasetsPage = () => {
  const [tableData, setTableData] = useState<IDataset[]>([])

  useEffect(() => {
    getDatasets()
      .then((response) => {
        console.log(response.data)
        setTableData(response.data)
      })
      .catch((e) => {
        console.log(e)
      })
  }, [setTableData])

  return (
    <Box p={4} pt={0} width='100%'>
      <Box display='flex' justifyContent='space-between'>
        <Text fontSize="2xl" mb={4}>Датасеты</Text>
        <Button variant="surface" onClick={() => window.open('http://172.16.1.10:8080/projects/create', '_blank')}>Новый датасет</Button>
      </Box>

      <Table.Root variant="outline">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader>ID</Table.ColumnHeader>
            <Table.ColumnHeader>Название</Table.ColumnHeader>
            <Table.ColumnHeader>Дата создания</Table.ColumnHeader>
            <Table.ColumnHeader>Статус</Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {tableData
            ? tableData?.map?.((data: IDataset) => {
              return (
                <Table.Row
                  cursor="pointer"
                  _hover={{
                    background: 'gray.100',
                  }}
                >
                  <Table.Cell>{data.id}</Table.Cell>
                  <Table.Cell>{data.name}</Table.Cell>
                  <Table.Cell>{getDate(data.created_date)}</Table.Cell>
                  <Table.Cell>{data.status}</Table.Cell>
                </Table.Row>
              )
            })
            : null}

        </Table.Body>
      </Table.Root>
    </Box >
  )
}

export default DatasetsPage
