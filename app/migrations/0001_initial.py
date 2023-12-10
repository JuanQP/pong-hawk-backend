# Generated by Django 4.2.8 on 2023-12-10 01:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Game',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField()),
                ('video', models.FileField(upload_to='')),
                ('processed_video', models.FileField(null=True, upload_to='')),
                ('heatmap', models.FileField(null=True, upload_to='')),
                ('status', models.CharField(choices=[('FI', 'Finished'), ('PE', 'Pending'), ('PR', 'Processing')], default='PE', max_length=2)),
            ],
        ),
    ]