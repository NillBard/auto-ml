import { lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Flex } from '@chakra-ui/react'
import Menu from '../components/commonMenu'
import Main from './main'

const PipelinePage = lazy(() => import('./createPipeline'))
const TablePage = lazy(() => import('./historyTable'))

export default function Routing() {
  return (
    <Flex>
      <Menu />
      <Routes>
        <Route path={'/'} element={<Main />} />
        <Route path={'/table'} element={<TablePage />} />
        <Route path={'/pipeline'} element={<PipelinePage />} />
      </Routes>
    </Flex>
  )
}
