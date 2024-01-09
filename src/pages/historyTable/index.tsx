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
} from '@chakra-ui/react'

interface ITrainingData {
  id: number
  name: string
  model: string
  task: string
  createdAt: string
}

const HistoryTable = () => {
  const tableData = [
    {
      id: 0,
      name: 'name',
      model: 'yolov4',
      task: 'object detection',
      createdAt: '18/04/2012 15:07:33',
    },
    {
      id: 1,
      name: 'name1',
      model: 'yolov5',
      task: 'image classification',
      createdAt: '18/04/2012 15:07:33',
    },
    {
      id: 2,
      name: 'name2',
      model: 'yolov8',
      task: 'image segmentation',
      createdAt: '18/04/2012 15:07:33',
    },
    {
      id: 3,
      name: 'name3',
      model: 'yolov4',
      task: 'object detection',
      createdAt: '18/04/2012 15:07:33',
    },
  ]

  return (
    <Flex width="100%">
      <TableContainer width="100%" pl="100px" pt="50px">
        <Table variant="simple">
          <Thead>
            <Tr>
              <Th>ID</Th>
              <Th>Название</Th>
              <Th>Модель</Th>
              <Th>Задача</Th>
              <Th>Дата создания</Th>
              <Th>Изменить</Th>
            </Tr>
          </Thead>
          <Tbody>
            {tableData.map((data: ITrainingData) => {
              return (
                <Tr>
                  <Td>{data.id}</Td>
                  <Td>{data.name}</Td>
                  <Td>{data.model}</Td>
                  <Td>{data.task}</Td>
                  <Td>{data.createdAt}</Td>
                  <Td>
                    <Link>Смотреть конфигурацию</Link>
                  </Td>
                </Tr>
              )
            })}
          </Tbody>
        </Table>
      </TableContainer>
    </Flex>
  )
}

export default HistoryTable
