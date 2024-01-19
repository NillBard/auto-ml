import {
  Table,
  Th,
  Thead,
  Tr,
  Flex,
  TableContainer,
  Tbody,
  Td,
  Link,
  Button,
} from '@chakra-ui/react'

import { getTrainingConfigurations } from '../../api/api.ts'
import { useEffect, useState } from 'react'
import { ITrain } from '../../types/train.ts'

const HistoryTable = () => {
  const [tableData, setTableData] = useState<[ITrain]>()

  useEffect(() => {
    const getConf = () => {
      getTrainingConfigurations()
        .then((response) => {
          console.log(response.data)
          setTableData(response.data)
        })
        .catch((e) => {
          console.log(e)
        })
    }
    getConf()
  }, [])
  return (
    <Flex width="100%" direction="column" gap="10px">
      <TableContainer width="100%" pl="100px" pt="50px">
        <Table variant="simple">
          <Thead>
            <Tr>
              <Th>ID</Th>
              <Th>Название</Th>
              <Th>Модель</Th>
              <Th>Дата создания</Th>
              <Th>Статус</Th>
              <Th>Конфигурация</Th>
            </Tr>
          </Thead>
          <Tbody>
            {tableData
              ? tableData.map((data: ITrain) => {
                  return (
                    <Tr>
                      <Td>{data.id}</Td>
                      <Td>{data.name}</Td>
                      <Td>{data.model}</Td>
                      <Td>{data.created_at}</Td>
                      <Td>{data.status}</Td>
                      <Td>
                        <Link>Конфигурация</Link>
                      </Td>
                    </Tr>
                  )
                })
              : null}
          </Tbody>
        </Table>
      </TableContainer>
      <Button w="400px" alignSelf="center">
        Новая конфигурация
      </Button>
    </Flex>
  )
}

export default HistoryTable
