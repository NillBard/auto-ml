import { Table, Thead } from '@chakra-ui/react'

// import tableData from '../../lib/tableData.ts'

const HistoryTable = () => {
  return (
    <Table>
      <Thead>
        <Tr>
          <Th>ID</Th>
          <Th>Название</Th>
          <Th>Модель</Th>
          <Th>Задача</Th>
          <Th>Дата создания</Th>
        </Tr>
      </Thead>
    </Table>
  )
}

export default HistoryTable
