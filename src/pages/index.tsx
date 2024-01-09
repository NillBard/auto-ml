import { lazy } from 'react'
import { Route, Routes } from 'react-router-dom'

const PipelinePage = lazy(() => import('./createPipeline'))
const TablePage = lazy(() => import('./historyTable'))

export default function Routing() {
  return (
    <>
      <Routes>
        <Route path={'/table'} element={<TablePage />} />
        <Route path={'/pipeline'} element={<PipelinePage />} />
      </Routes>
    </>
  )
}
