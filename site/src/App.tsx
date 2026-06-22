import { ProgressBar } from './components/ProgressBar'
import { Header } from './components/Header'
import { Footer } from './components/Footer'
import { Hero } from './sections/Hero'
import { Questions } from './sections/Questions'
import { ProblemStatement } from './sections/ProblemStatement'
import { ExampleGallery } from './components/ExampleGallery'
import { Disclosure } from './components/Disclosure'
import { RelatedWork } from './sections/RelatedWork'
import { SystemDesign } from './sections/SystemDesign'
import { ModelMRI } from './sections/ModelMRI'
import { DatasetsAtAGlance } from './sections/DatasetsAtAGlance'
import { Ablations } from './sections/Ablations'
import { Proof } from './sections/Proof'
import { Conclusion } from './sections/Conclusion'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <ProgressBar />
      <Header />
      <main>
        <Hero />
        <Questions />
        <ProblemStatement />
        <Disclosure label="Show real-frame examples">
          <ExampleGallery />
        </Disclosure>
        <RelatedWork />
        <SystemDesign />
        <ModelMRI />
        <DatasetsAtAGlance />
        <Ablations />
        <Proof />
        <Conclusion />
      </main>
      <Footer />
    </div>
  )
}
