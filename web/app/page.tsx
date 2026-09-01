export default function Home() {
  return (
    <main className="console-shell">
      <header className="site-header">
        <a className="wordmark" href="/" aria-label="Контур — главная">КОНТУР</a>
        <span className="header-label">Инспекция стройплощадки</span>
        <span className="header-status">Система готова</span>
      </header>

      <section className="welcome-panel" aria-labelledby="welcome-title">
        <p className="eyebrow">НОВАЯ ПРОВЕРКА</p>
        <h1 id="welcome-title">Загрузите запись с объекта</h1>
        <p>Видео или фото будут проверены на соблюдение требований безопасности и ход работ.</p>
        <p className="panel-note">Загрузка и запуск анализа появятся на следующем этапе.</p>
      </section>
    </main>
  )
}
