import { lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Flex } from '@chakra-ui/react'

const LoginPage = lazy(() => import('./Login.tsx'))
const MainPage = lazy(() => import('./Main.tsx'))
const DatasetsPage = lazy(() => import('./Datasets.tsx'))
const ProcessingsPage = lazy(() => import('./Processings.tsx'))
const TrainingPage = lazy(() => import('./Training.tsx'))
const ConfigurePage = lazy(() => import('./Configure.tsx'))
const NewConfigurePage = lazy(() => import('./NewConfigure.tsx'))

// const PipelinePage = lazy(() => import('./createPipeline'))
const PipelinePage = lazy(() => import('./Pipeline.tsx'))

import { CommonMenu, CommonLayout } from '@/shared/ui'

export default function Routing() {
  return (
    <Flex>
      <CommonMenu />
      <CommonLayout>
        <Routes>
          <Route path="/" element={<LoginPage />}></Route>
          <Route path="/main" element={<MainPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/pipeline" element={<PipelinePage />} />
          <Route path="/training/configuration" element={<NewConfigurePage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/processings" element={<ProcessingsPage />} />
          <Route
            path="/training/configuration/:configurationId"
            element={<ConfigurePage />}
          />
        </Routes>
      </CommonLayout>
    </Flex>
  )
}
