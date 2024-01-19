import { lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Flex } from '@chakra-ui/react'
import Menu from '../components/commonMenu'

const PipelinePage = lazy(() => import('./createPipeline'))
const TablePage = lazy(() => import('./historyTable'))
const MainPage = lazy(() => import('./main'))
const ConfigurePage = lazy(() => import('./configureTraining'))

export default function Routing() {
  return (
    <Flex>
      <Menu />
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/training" element={<TablePage />} />
        <Route path="/pipeline" element={<PipelinePage />} />
        <Route path="/training/configure" element={<ConfigurePage />} />
      </Routes>
    </Flex>
  )
}
